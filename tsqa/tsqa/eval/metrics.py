from collections import defaultdict


def compute_metrics(results: list) -> dict:
    """
    Compute accuracy metrics from a list of pipeline result dicts.

    Args:
        results: list of dicts returned by run_pipeline / run_dataset

    Returns dict with keys:
        overall_accuracy:   float  correct / total (excluding rows with no gold label)
        total:              int    total rows evaluated
        correct:            int    number of correct predictions
        no_gold:            int    rows skipped (no parseable gold label)
        parse_failures:     int    rows where the answer parser returned None
        fallback_rate:      float  fraction of rows that used the fallback path
        by_category:        dict   {category: {accuracy, correct, total}}
        by_branch:          dict   {branch_used: {accuracy, correct, total}}
        by_flags:           dict   {flag: {accuracy, correct, total}}
                            accuracy for rows that had this flag present
    """
    total       = 0
    correct     = 0
    no_gold     = 0
    parse_fails = 0
    fallbacks   = 0

    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    by_branch   = defaultdict(lambda: {"correct": 0, "total": 0})
    by_flags    = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        gold   = r.get("gold_letter")
        pred   = r.get("predicted_letter")
        cat    = r.get("category", "unknown") or "unknown"
        branch = r.get("branch_used", "fallback") or "fallback"
        flags  = r.get("flags") or []
        is_correct = r.get("correct", False)

        if gold is None:
            no_gold += 1
            continue

        total += 1
        if pred is None:
            parse_fails += 1
        if r.get("used_fallback"):
            fallbacks += 1
        if is_correct:
            correct += 1

        # per-category
        by_category[cat]["total"]   += 1
        by_category[cat]["correct"] += int(is_correct)

        # per-branch
        by_branch[branch]["total"]   += 1
        by_branch[branch]["correct"] += int(is_correct)

        # per-flag (a row can contribute to multiple flag buckets)
        for flag in flags:
            by_flags[flag]["total"]   += 1
            by_flags[flag]["correct"] += int(is_correct)

    def _acc(d):
        return round(d["correct"] / d["total"], 4) if d["total"] > 0 else None

    return {
        "overall_accuracy": round(correct / total, 4) if total > 0 else None,
        "total":            total,
        "correct":          correct,
        "no_gold":          no_gold,
        "parse_failures":   parse_fails,
        "fallback_rate":    round(fallbacks / total, 4) if total > 0 else None,
        "by_category": {
            cat: {**v, "accuracy": _acc(v)}
            for cat, v in by_category.items()
        },
        "by_branch": {
            b: {**v, "accuracy": _acc(v)}
            for b, v in by_branch.items()
        },
        "by_flags": {
            f: {**v, "accuracy": _acc(v)}
            for f, v in by_flags.items()
        },
    }
