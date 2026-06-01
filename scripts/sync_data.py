"""Git-style bidirectional sync between local data/ and the HF backup repo.

Plain `hf download` and `hf upload` are additive and one-directional. This
script reconciles both sides the way git does, with a three-way comparison
between the current local tree, the current remote tree, and a manifest of the
last synced state. The manifest is what tells the two sides apart. Without a
common base, a local delete looks identical to a remote add. For each file the
script decides one of pull, push, delete-here, delete-there, rename, conflict,
or skip, then applies the plan.

Remote state is read with list_repo_tree, which walks the git tree instead of
resolving every blob. snapshot_download resolves all files and trips the HF
resolver limit of 5000 requests per 5 minutes on a repo with thousands of small
files, which is the failure this design avoids. Only files that genuinely
differ are transferred.

Content identity uses the same hashes HF stores, sha256 for LFS files and the
git blob sha1 for the rest, so local and remote compare exactly without a
download. A stat cache in the manifest skips re-hashing files whose size and
mtime have not moved.

On the very first run there is no manifest, so the reconcile is purely additive.
Files missing on one side are copied to it and files that differ are reported as
conflicts. Nothing is deleted until a manifest exists to prove the file was
synced before and then removed.

Usage.
    python scripts/sync_data.py            # reconcile both directions
    python scripts/sync_data.py --dry-run  # print the plan, change nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

REPO = "wahyyuht/skripsi-data"
LOCAL_DIR = Path("data")
MANIFEST_PATH = LOCAL_DIR / ".sync_manifest.json"
_NEW = "<new>"  # sentinel hash for a local-only file, never equals a real hash


@dataclass
class RemoteFile:
    """A file in the HF repo identified by the content hash HF stores for it."""

    hash: str  # lfs sha256 when stored in lfs, otherwise the git blob sha1
    lfs: bool


@dataclass
class Entry:
    """One manifest record, the agreed state of a file at the last sync."""

    hash: str
    lfs: bool
    size: int
    mtime_ns: int


@dataclass
class Plan:
    """The set of actions that makes local and remote agree."""

    pull: list[str]
    push: list[str]
    del_local: list[str]
    del_remote: list[str]
    renames: list[tuple[str, str]]  # (from_path, to_path), a local move
    conflicts: list[str]


def git_blob_sha1(path: Path) -> str:
    """Return the git blob sha1 of a file, matching HF blob ids for non-LFS files."""
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of(path: Path) -> str:
    """Return the sha256 of a file's content, matching HF LFS sha256 ids."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_hash(path: Path, lfs: bool) -> str:
    """Hash a local file the way HF would so it compares equal to the remote id."""
    return sha256_of(path) if lfs else git_blob_sha1(path)


def _ignored(rel: str) -> bool:
    """Skip our own manifest and the HF download cache that hf_hub_download writes."""
    return (
        rel == MANIFEST_PATH.name
        or rel.startswith(".git")
        or rel.startswith(".cache/huggingface")
    )


def list_local(root: Path) -> set[str]:
    """Return relative POSIX paths of every file under root, minus sync metadata."""
    if not root.exists():
        return set()
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and not _ignored(p.relative_to(root).as_posix())
    }


def scan_remote(api: HfApi) -> dict[str, RemoteFile]:
    """List the remote git tree and return each file's HF content hash.

    Reads the tree without resolving blobs, so it does not hit the resolver
    rate limit the way snapshot_download does.
    """
    out: dict[str, RemoteFile] = {}
    for item in api.list_repo_tree(REPO, repo_type="dataset", recursive=True):
        if not hasattr(item, "blob_id"):  # folders carry no blob id
            continue
        if _ignored(item.path):
            continue
        if item.lfs is not None:
            out[item.path] = RemoteFile(item.lfs.sha256, True)
        else:
            out[item.path] = RemoteFile(item.blob_id, False)
    return out


def load_manifest() -> dict[str, Entry]:
    """Load the last-synced manifest, or an empty one on first run."""
    if not MANIFEST_PATH.exists():
        return {}
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {p: Entry(**e) for p, e in raw.items()}


def save_manifest(entries: dict[str, Entry]) -> None:
    """Write the manifest of the now-consistent state, compact and key-sorted."""
    data = {p: vars(e) for p, e in entries.items()}
    MANIFEST_PATH.write_text(
        json.dumps(data, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )


def current_local_hash(path: Path, lfs: bool, base: Entry | None) -> str:
    """Hash a local file, reusing the manifest hash when size and mtime match."""
    st = path.stat()
    if base is not None and base.size == st.st_size and base.mtime_ns == st.st_mtime_ns:
        return base.hash
    return hf_hash(path, lfs)


def reconcile(
    root: Path, remote: dict[str, RemoteFile], manifest: dict[str, Entry]
) -> tuple[Plan, dict[str, str]]:
    """Compare local, remote, and the base manifest into an action plan.

    Also returns the local hash computed for each path that needed one, so the
    caller can detect renames and rebuild the manifest without hashing twice.
    """
    local = list_local(root)
    plan = Plan([], [], [], [], [], [])
    local_hashes: dict[str, str] = {}

    for p in sorted(local | set(remote) | set(manifest)):
        rem = remote.get(p)
        base = manifest.get(p)
        lfs = rem.lfs if rem else (base.lfs if base else False)

        if p in local and (rem is not None or base is not None):
            h_local = current_local_hash(root / p, lfs, base)
            local_hashes[p] = h_local
        elif p in local:
            h_local = _NEW  # new local-only file, no comparison needed
        else:
            h_local = None
        h_remote = rem.hash if rem else None
        h_base = base.hash if base else None

        if h_local == h_remote:  # already identical, covers both-absent
            continue

        local_changed = h_local != h_base
        remote_changed = h_remote != h_base

        if local_changed and remote_changed:
            plan.conflicts.append(p)
        elif local_changed:
            (plan.push if p in local else plan.del_remote).append(p)
        elif remote_changed:
            (plan.pull if rem is not None else plan.del_local).append(p)

    _detect_renames(plan, remote, local_hashes)
    return plan, local_hashes


def _detect_renames(
    plan: Plan, remote: dict[str, RemoteFile], local_hashes: dict[str, str]
) -> None:
    """Fold a delete-here plus download-identical-content into a local move."""
    by_hash: dict[str, list[str]] = {}
    for p in plan.del_local:
        by_hash.setdefault(local_hashes[p], []).append(p)
    for tgt in list(plan.pull):
        sources = by_hash.get(remote[tgt].hash)
        if sources:
            src = sources.pop()
            plan.renames.append((src, tgt))
            plan.pull.remove(tgt)
            plan.del_local.remove(src)


def print_plan(plan: Plan) -> None:
    """Print a one-line-per-group summary with a short preview of each group."""
    groups = [
        ("pull", plan.pull),
        ("push", plan.push),
        ("delete local", plan.del_local),
        ("delete remote", plan.del_remote),
        ("rename", [f"{s} -> {t}" for s, t in plan.renames]),
        ("conflict", plan.conflicts),
    ]
    if not any(items for _, items in groups):
        print("already in sync")
        return
    for name, items in groups:
        if not items:
            continue
        print(f"{name}: {len(items)}")
        for line in items[:10]:
            print(f"  {line}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")


def prune_empty_dirs(root: Path) -> None:
    """Remove directories left empty after deletions, deepest first."""
    for d in sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda x: len(x.parts),
        reverse=True,
    ):
        try:
            d.rmdir()
        except OSError:
            pass


def apply_plan(api: HfApi, root: Path, plan: Plan) -> None:
    """Apply moves, local deletes, downloads, then one atomic remote commit."""
    for src, tgt in plan.renames:
        dest = root / tgt
        dest.parent.mkdir(parents=True, exist_ok=True)
        (root / src).replace(dest)

    for p in plan.del_local:
        (root / p).unlink(missing_ok=True)
    if plan.del_local or plan.renames:
        prune_empty_dirs(root)

    for i, p in enumerate(plan.pull, 1):
        hf_hub_download(REPO, repo_type="dataset", filename=p, local_dir=str(root))
        if i % 25 == 0 or i == len(plan.pull):
            print(f"  downloaded {i}/{len(plan.pull)}")

    ops = [
        CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(root / p))
        for p in plan.push
    ]
    ops += [CommitOperationDelete(path_in_repo=p) for p in plan.del_remote]
    if ops:
        api.create_commit(
            REPO,
            repo_type="dataset",
            operations=ops,
            commit_message="sync: reconcile local and remote",
        )
        print(f"  committed {len(plan.push)} adds, {len(plan.del_remote)} deletes")


def rebuild_manifest(
    api: HfApi, root: Path, old: dict[str, Entry], conflicts: list[str]
) -> dict[str, Entry]:
    """Snapshot the now-consistent state, carrying unresolved conflicts forward."""
    entries: dict[str, Entry] = {}
    for p, rf in scan_remote(api).items():
        fp = root / p
        if not fp.exists():
            continue
        st = fp.stat()
        entries[p] = Entry(rf.hash, rf.lfs, st.st_size, st.st_mtime_ns)
    for p in conflicts:
        if p in old:
            entries[p] = old[p]
    return entries


def main() -> int:
    """Reconcile both directions, or print the plan under --dry-run."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--dry-run", action="store_true", help="Print the plan without changing anything."
    )
    args = ap.parse_args()

    api = HfApi()
    remote = scan_remote(api)
    manifest = load_manifest()
    plan, _ = reconcile(LOCAL_DIR, remote, manifest)
    print_plan(plan)

    if args.dry_run:
        return 0

    if plan.conflicts:
        print("\nconflicts changed on both sides and were left untouched, resolve manually")

    apply_plan(api, LOCAL_DIR, plan)
    save_manifest(rebuild_manifest(api, LOCAL_DIR, manifest, plan.conflicts))
    print("sync complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
