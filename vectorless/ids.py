"""Document-id parsing utilities shared by indexing and retrieval."""
from .categories import CATEGORIES


def _build_prefix_table() -> list[tuple[str, str]]:
    """Return (prefix, folder) pairs sorted longest prefix first."""
    table = [(c.doc_id_prefix(), c.folder) for c in CATEGORIES]
    return sorted(table, key=lambda pair: -len(pair[0]))


_PREFIX_TABLE = _build_prefix_table()


def doc_category(doc_id: str) -> str:
    """Map a doc_id to its category folder (e.g. "uu-1-2025" to "UU").

    Raises ValueError if no registered Category prefix matches.
    """
    low = doc_id.lower()
    for prefix, folder in _PREFIX_TABLE:
        if low.startswith(prefix + "-"):
            return folder
    raise ValueError(
        f"doc_id {doc_id!r} matches no registered Category prefix; "
        f"add an entry to vectorless.categories.CATEGORIES"
    )
