"""Statistical significance tests for between-system comparison.

Paired randomization is the primary test, with no normality assumption.
The paired t-test is the secondary. Effect size is Cohen's d for paired
samples and Cliff's delta for two independent samples. The module also provides
a percentile bootstrap confidence interval, a two-sample randomization test for
difference-of-differences contrasts, and the Holm-Bonferroni correction for a
family of tests.

Pure Python plus optional scipy. The randomization test never needs scipy.
The t-test prefers scipy.stats but falls back to a normal approximation.
"""

from __future__ import annotations

import math
import random

VALID_ALTERNATIVES = ("two-sided", "greater", "less")


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

DEFAULT_RANDOMIZATION_B = 10000
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_SEED = 42


# ----------------------------------------------------------------------
# Paired randomization test
# ----------------------------------------------------------------------

def paired_randomization(
    a: list[float],
    b: list[float],
    *,
    alternative: str = "two-sided",
    B: int = DEFAULT_RANDOMIZATION_B,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Paired randomization test for the mean of differences.

    For each query i compute d_i = a_i - b_i. Null hypothesis, the sign of
    d_i is exchangeable, equivalent to "system labels are interchangeable".
    Resample sign assignments B times and recompute the mean each time. For a
    two-sided test count how often the absolute permuted mean is at least as
    extreme as the observed absolute mean. For "greater" count permuted means
    at least the observed, for "less" count those at most the observed.
    """
    if alternative not in VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {VALID_ALTERNATIVES}, got {alternative!r}")
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be same length, got {len(a)} and {len(b)}")
    n = len(a)
    if n < 2:
        return {
            "method": "paired-randomization",
            "n": n, "B": B, "seed": seed, "alternative": alternative,
            "mean_diff": 0.0, "p_value": 1.0,
            "note": "n < 2, test undefined",
        }

    diffs = [float(ai) - float(bi) for ai, bi in zip(a, b)]
    observed = sum(diffs) / n
    abs_observed = abs(observed)

    rng = random.Random(seed)
    extreme_count = 0
    for _ in range(B):
        permuted_sum = 0.0
        for d in diffs:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            permuted_sum += sign * d
        permuted = permuted_sum / n
        if alternative == "two-sided":
            hit = abs(permuted) >= abs_observed
        elif alternative == "greater":
            hit = permuted >= observed
        else:
            hit = permuted <= observed
        if hit:
            extreme_count += 1

    # Add-one smoothed p-value, never zero.
    p_value = (extreme_count + 1) / (B + 1)
    return {
        "method": "paired-randomization",
        "n": n,
        "B": B,
        "seed": seed,
        "alternative": alternative,
        "mean_diff": observed,
        "p_value": p_value,
    }


# ----------------------------------------------------------------------
# Paired t-test (secondary test)
# ----------------------------------------------------------------------

def paired_t_test(a: list[float], b: list[float], *, alternative: str = "two-sided") -> dict:
    """Paired t-test on the mean of differences, two-sided by default."""
    if alternative not in VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {VALID_ALTERNATIVES}, got {alternative!r}")
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be same length, got {len(a)} and {len(b)}")
    n = len(a)
    if n < 2:
        return {
            "method": "paired-t-test",
            "n": n, "p_value": 1.0, "t_stat": 0.0, "df": 0,
            "mean_diff": 0.0, "std_err": 0.0, "alternative": alternative,
            "note": "n < 2, test undefined",
        }

    diffs = [float(ai) - float(bi) for ai, bi in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    df = n - 1
    if var_d == 0.0:
        return {
            "method": "paired-t-test",
            "n": n, "df": df,
            "t_stat": 0.0, "mean_diff": mean_d, "std_err": 0.0,
            "p_value": 1.0, "alternative": alternative,
            "note": "zero variance, all differences identical",
        }

    se = math.sqrt(var_d / n)
    t_stat = mean_d / se
    p_two = _two_sided_p_t(t_stat, df)
    if alternative == "two-sided":
        p_value = p_two
    elif alternative == "greater":
        p_value = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
    else:
        p_value = p_two / 2.0 if t_stat < 0 else 1.0 - p_two / 2.0
    return {
        "method": "paired-t-test",
        "n": n,
        "df": df,
        "t_stat": t_stat,
        "mean_diff": mean_d,
        "std_err": se,
        "alternative": alternative,
        "p_value": p_value,
    }


def _two_sided_p_t(t_stat: float, df: int) -> float:
    """Two-sided p-value for a t-statistic. Uses scipy if available."""
    try:
        from scipy.stats import t as scipy_t
        return float(2.0 * scipy_t.sf(abs(t_stat), df))
    except ImportError:
        # Normal approximation, fine for thesis-scale n.
        return 2.0 * (1.0 - _normal_cdf(abs(t_stat)))


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ----------------------------------------------------------------------
# Effect size, Cohen's d for paired samples
# ----------------------------------------------------------------------

def cohens_d_paired(a: list[float], b: list[float]) -> dict:
    """Cohen's d for paired samples, mean of differences over their SD."""
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be same length, got {len(a)} and {len(b)}")
    n = len(a)
    if n < 2:
        return {"d": 0.0, "label": "n/a", "n": n, "note": "n < 2"}
    diffs = [float(ai) - float(bi) for ai, bi in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var_d) if var_d > 0 else 0.0
    if sd == 0.0:
        return {"d": 0.0, "label": "zero-variance", "n": n}
    d = mean_d / sd
    return {"d": d, "label": sawilowsky_label(d), "n": n}


def sawilowsky_label(d: float) -> str:
    """Descriptive label for Cohen's d magnitude."""
    abs_d = abs(d)
    if abs_d < 0.01:
        return "trivial"
    if abs_d < 0.20:
        return "very small"
    if abs_d < 0.50:
        return "small"
    if abs_d < 0.80:
        return "medium"
    if abs_d < 1.20:
        return "large"
    if abs_d < 2.0:
        return "very large"
    return "huge"


# ----------------------------------------------------------------------
# Effect size, Cliff's delta for two independent samples
# ----------------------------------------------------------------------

def cliffs_delta(a: list[float], b: list[float]) -> dict:
    """Cliff's delta, a nonparametric effect size for two independent samples.

    Delta is the probability that a random value from a exceeds one from b minus
    the reverse, in the range -1 to 1. It is robust to the bounded, skewed shape
    of per-query retrieval metrics, where Cohen's d is distorted. Used for the
    H8 interaction, the subset margins against the complement margins. Magnitude
    thresholds follow Romano et al. 2006.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return {"delta": 0.0, "label": "n/a", "n_a": na, "n_b": nb}
    greater = less = 0
    for x in a:
        for y in b:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    delta = (greater - less) / (na * nb)
    return {"delta": delta, "label": cliffs_label(delta), "n_a": na, "n_b": nb}


def cliffs_label(delta: float) -> str:
    """Magnitude label for Cliff's delta, Romano et al. 2006 thresholds."""
    abs_d = abs(delta)
    if abs_d < 0.147:
        return "negligible"
    if abs_d < 0.33:
        return "small"
    if abs_d < 0.474:
        return "medium"
    return "large"


# ----------------------------------------------------------------------
# Percentile bootstrap confidence interval
# ----------------------------------------------------------------------

def bootstrap_ci_paired(
    a: list[float],
    b: list[float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Percentile bootstrap CI on the mean paired difference a - b.

    Resamples the paired differences with replacement, which keeps the
    pairing intact, and takes the empirical percentiles of the resampled
    means.
    """
    if len(a) != len(b):
        raise ValueError(f"paired arrays must be same length, got {len(a)} and {len(b)}")
    n = len(a)
    diffs = [float(ai) - float(bi) for ai, bi in zip(a, b)]
    if n < 2:
        point = diffs[0] if diffs else 0.0
        return {"low": point, "high": point, "confidence": confidence, "n": n}

    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[int(tail * resamples)]
    high = means[min(int((1.0 - tail) * resamples), resamples - 1)]
    return {"low": low, "high": high, "confidence": confidence, "n": n}


# ----------------------------------------------------------------------
# Two-sample randomization for a difference-of-differences contrast
# ----------------------------------------------------------------------

def two_sample_randomization(
    values: list[float],
    n_subset: int,
    *,
    alternative: str = "greater",
    B: int = DEFAULT_RANDOMIZATION_B,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Test whether a subset mean exceeds the complement mean.

    The input is one per-query value for every query, with the first
    n_subset entries forming the subset of interest. The statistic is the
    subset mean minus the complement mean. The null shuffles the subset label
    across queries. Used for the multihop difference-of-differences contrast,
    where each value is a per-query paradigm margin.
    """
    if alternative not in VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {VALID_ALTERNATIVES}, got {alternative!r}")
    n = len(values)
    if not 0 < n_subset < n:
        raise ValueError(f"n_subset must be between 1 and {n - 1}, got {n_subset}")

    def contrast(order: list[float]) -> float:
        sub = sum(order[:n_subset]) / n_subset
        comp = sum(order[n_subset:]) / (n - n_subset)
        return sub - comp

    observed = contrast(values)
    rng = random.Random(seed)
    extreme_count = 0
    for _ in range(B):
        picked = rng.sample(range(n), n_subset)
        picked_set = set(picked)
        sub = sum(values[i] for i in picked) / n_subset
        comp = sum(values[i] for i in range(n) if i not in picked_set) / (n - n_subset)
        stat = sub - comp
        if alternative == "two-sided":
            hit = abs(stat) >= abs(observed)
        elif alternative == "greater":
            hit = stat >= observed
        else:
            hit = stat <= observed
        if hit:
            extreme_count += 1

    return {
        "method": "two-sample-randomization",
        "n": n,
        "n_subset": n_subset,
        "B": B,
        "seed": seed,
        "alternative": alternative,
        "contrast": observed,
        "p_value": (extreme_count + 1) / (B + 1),
    }


def two_sample_bootstrap_ci(
    subset: list[float],
    complement: list[float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Percentile bootstrap CI on the subset-minus-complement contrast.

    Resamples within each group independently, matching the structure of the
    two-sample randomization test.
    """
    rng = random.Random(seed)
    ns, nc = len(subset), len(complement)
    contrasts = []
    for _ in range(resamples):
        sub = sum(subset[rng.randrange(ns)] for _ in range(ns)) / ns
        comp = sum(complement[rng.randrange(nc)] for _ in range(nc)) / nc
        contrasts.append(sub - comp)
    contrasts.sort()
    tail = (1.0 - confidence) / 2.0
    low = contrasts[int(tail * resamples)]
    high = contrasts[min(int((1.0 - tail) * resamples), resamples - 1)]
    return {"low": low, "high": high, "confidence": confidence}


# ----------------------------------------------------------------------
# Multiple-comparison correction
# ----------------------------------------------------------------------

def holm_bonferroni(pvalues: dict[str, float], *, alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down adjustment for a family of p-values.

    Returns, per key, the adjusted p-value and whether it rejects the null at
    the family-wise level alpha. Adjusted values are monotone in rank, as the
    step-down procedure requires.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adjusted = min((m - i) * p, 1.0)
        running = max(running, adjusted)
        out[name] = {"p_raw": p, "p_adjusted": running, "reject": running < alpha}
    return out


# ----------------------------------------------------------------------
# Convenience, run the whole headline suite at once
# ----------------------------------------------------------------------

def compare_paired(
    a: list[float],
    b: list[float],
    *,
    alternative: str = "two-sided",
    B: int = DEFAULT_RANDOMIZATION_B,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Bundle randomization, t-test, Cohen's d, and a bootstrap CI into one dict.

    Use this as the headline output for one (system_a, system_b, metric). The
    alternative applies to both the randomization test and the t-test.
    """
    rand = paired_randomization(a, b, alternative=alternative, B=B, seed=seed)
    t = paired_t_test(a, b, alternative=alternative)
    eff = cohens_d_paired(a, b)
    ci = bootstrap_ci_paired(a, b, resamples=bootstrap_resamples, seed=seed)
    convergent = (
        rand.get("p_value", 1.0) < 0.05
    ) == (
        t.get("p_value", 1.0) < 0.05
    )
    return {
        "n": rand.get("n", len(a)),
        "mean_diff": rand.get("mean_diff", 0.0),
        "alternative": alternative,
        "paired_randomization": rand,
        "paired_t_test": t,
        "cohens_d": eff,
        "bootstrap_ci": ci,
        "tests_converge": convergent,
    }


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def run_self_test() -> None:
    """Sanity tests for each function. Called from the eval CLI on demand."""
    def assert_close(actual: float, expected: float, tol: float = 1e-6) -> None:
        """Raise AssertionError unless actual equals expected within tol."""
        if abs(actual - expected) > tol:
            raise AssertionError(f"expected {expected}, got {actual}")

    # Identical inputs -> p approx 1, mean_diff 0
    a = [1.0, 0.0, 1.0, 1.0, 0.0] * 20  # n=100
    res = compare_paired(a, a, B=2000, seed=1)
    assert_close(res["mean_diff"], 0.0)
    if res["paired_randomization"]["p_value"] < 0.5:
        raise AssertionError(
            f"identical inputs should give large p, got {res['paired_randomization']['p_value']}"
        )

    # Obvious difference -> small p
    b_better = [1.0] * 100
    b_worse = [0.0] * 100
    res = compare_paired(b_better, b_worse, B=2000, seed=1)
    if res["paired_randomization"]["p_value"] >= 0.01:
        raise AssertionError(
            f"obvious diff should give small p, got {res['paired_randomization']['p_value']}"
        )
    if res["cohens_d"]["d"] != 0 and res["cohens_d"]["label"] != "zero-variance":
        # All-1 vs all-0 has zero variance of differences (all diffs = 1).
        # That is the documented edge-case path.
        raise AssertionError(f"unexpected cohens_d branch, got {res['cohens_d']}")

    # Mild difference, n=200, mean diff 0.05
    rng = random.Random(7)
    s_a = [1.0 if rng.random() < 0.55 else 0.0 for _ in range(200)]
    s_b = [1.0 if rng.random() < 0.50 else 0.0 for _ in range(200)]
    res = compare_paired(s_a, s_b, B=5000, seed=1)
    # Don't assert p threshold (random noise), just sanity-check structure
    expected_keys = {"paired_randomization", "paired_t_test", "cohens_d", "tests_converge"}
    if not expected_keys.issubset(res.keys()):
        raise AssertionError(f"compare_paired missing keys, got {res.keys()}")

    # Sawilowsky label boundaries
    if sawilowsky_label(0.0) != "trivial":
        raise AssertionError("sawilowsky_label(0.0) wrong")
    if sawilowsky_label(0.30) != "small":
        raise AssertionError("sawilowsky_label(0.30) wrong")
    if sawilowsky_label(1.5) != "very large":
        raise AssertionError("sawilowsky_label(1.5) wrong")
    if sawilowsky_label(-2.5) != "huge":
        raise AssertionError("sawilowsky_label should use abs value")

    # Cliff's delta, full separation is +/-1, identical groups is 0
    if cliffs_delta([1.0, 1.0, 1.0], [0.0, 0.0])["delta"] != 1.0:
        raise AssertionError("cliffs_delta full separation should be 1.0")
    if cliffs_delta([0.0, 0.0], [1.0, 1.0, 1.0])["delta"] != -1.0:
        raise AssertionError("cliffs_delta reversed separation should be -1.0")
    if cliffs_delta([1.0, 0.0], [1.0, 0.0])["delta"] != 0.0:
        raise AssertionError("cliffs_delta identical groups should be 0.0")
    if cliffs_label(0.10) != "negligible" or cliffs_label(0.40) != "medium":
        raise AssertionError("cliffs_label thresholds wrong")
