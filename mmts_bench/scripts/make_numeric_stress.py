#!/usr/bin/env python3
# scripts/make_numeric_stress.py
"""
Synthetic numeric-PARSER stress set (the three-arm study, log/012).
===================================================================
MMTS-Base is too clean to expose the parser's brittleness (4 templates, 0
compositional / typo / paraphrase rows), and no real benchmark exists. So we
synthesize one — but in a way that isolates PARSING from computation and dodges
the "questions built for our parser" trap:

  * GOLD is computed by the SAME tools the arms use — `evaluate_plan(intended
    plan, series)`. A correct parse therefore yields an EXACT match; a wrong parse
    is wrong. The metric measures only whether each arm parsed to the right plan.
  * Series are REAL (reused from MMTS-Base), so the computation is on real data.
  * PARAPHRASES are authored by a held-out, cross-family LLM (gpt-4o-mini) with
    A1's synonyms BANNED, so the paraphrase layer is genuinely outside our grammar
    — the only place the A2 LLM-planner can earn its keep.

Layers: clean (control) · compositional · typo · paraphrase. Output CSV columns:
``sample_id, layer, intent_plan(json), question, ground_truth, series(json)``.

Usage
-----
    python scripts/make_numeric_stress.py --out outputs/numeric_stress.csv \
        --n-series 40 --seed 0            # paraphrases need OPENAI_API_KEY
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_AGENTIC = _ROOT.parent / "tsqa"
sys.path.insert(0, str(_AGENTIC))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT.parent / ".env")
except Exception:
    pass

from tsqa.eval.numeric_head import evaluate_plan  # gold == the tools' own answer

# Quantity -> the ladder-natural phrase A1's grammar already recognises.
_QPHRASE = {"mean": "mean value", "var": "variance", "std": "standard deviation",
            "median": "median", "min": "minimum value", "max": "maximum value",
            "range": "range"}
# Natural composition surface forms (all A1-grammar-tractable on purpose: the point
# is that composition STRUCTURE is deterministically parseable — the LLM's value is
# the paraphrase layer, not this one).
_COMPOSE_FORMS = [
    ("subtract", "the {a} minus the {b}"),
    ("subtract", "the difference between the {a} and the {b}"),
    ("add", "the sum of the {a} and the {b}"),
    ("ratio", "the ratio of the {a} to the {b}"),
]
# Single-token typos in the quantity word (exercise A1's difflib + A2).
_TYPOS = {"standard deviation": "standar deviaton", "variance": "varaince",
          "median": "medain", "minimum value": "minimm value",
          "maximum value": "maximumm value", "mean value": "maen value"}
# Words a paraphrase of each quantity may NOT use → forces OPEN phrasings the A1
# synonym table cannot match (so A2's marginal value is real, not grammar-shaped).
_BANNED = {
    "std": ["standard", "deviation", "spread", "dispersion", "variability", "std",
            "variable", "how variable", "spread out"],
    "mean": ["mean", "average", "typical value", "central tendency", "central value"],
    "median": ["median", "middle value"],
    "min": ["minimum", "smallest", "lowest", "min"],
    "max": ["maximum", "largest", "highest", "peak", "max"],
    "range": ["range"],
}


def load_base_series(n, seed):
    f = _ROOT / "MMTS-BENCH" / "Benchmark" / "Base" / "synthtic_qa_700.csv"
    rng = random.Random(seed)
    out = []
    for r in csv.DictReader(open(f)):
        v = r.get("value")
        if not v or not v.strip().startswith("["):
            continue
        try:
            arr = [float(x) for x in json.loads(v)]
        except Exception:
            continue
        if len(arr) >= 16 and str(r.get("is_univariate", "True")) == "True":
            out.append(arr)
    rng.shuffle(out)
    return out[:n]


def _leaf(q):
    return {"quantity": q, "param": None, "series": "primary"}


def _gold(plan, ts):
    v, _ = evaluate_plan(plan, ts, None, None, {})
    return v


def gen_clean(series, rng):
    rows = []
    for i, ts in enumerate(series):
        q = rng.choice(list(_QPHRASE))
        g = _gold(_leaf(q), ts)
        if g is None:
            continue
        rows.append(("clean", json.dumps(_leaf(q)),
                     f"Based on the Time Series 1 data, what is the {_QPHRASE[q]} "
                     f"of the sequence?", g, ts))
    return rows


def gen_compositional(series, rng):
    rows, qs = [], list(_QPHRASE)
    for ts in series:
        op, form = rng.choice(_COMPOSE_FORMS)
        a, b = rng.sample(qs, 2)
        plan = {"op": op, "args": [_leaf(a), _leaf(b)], "series": "primary"}
        g = _gold(plan, ts)
        if g is None:
            continue
        phrase = form.format(a=_QPHRASE[a], b=_QPHRASE[b])
        rows.append(("compositional", json.dumps(plan),
                     f"Based on the Time Series 1 data, what is {phrase} of the "
                     f"sequence?", g, ts))
    return rows


def gen_typo(series, rng):
    rows = []
    phrase_to_q = {v: k for k, v in _QPHRASE.items()}
    for ts in series:
        good = rng.choice(list(_TYPOS))            # a clean phrase
        bad = _TYPOS[good]                          # its typo'd form
        q = phrase_to_q[good]
        g = _gold(_leaf(q), ts)
        if g is None:
            continue
        rows.append(("typo", json.dumps(_leaf(q)),
                     f"Based on the Time Series 1 data, what is the {bad} of the "
                     f"sequence?", g, ts))
    return rows


def gen_paraphrase(series, rng, n_per=1):
    """Held-out cross-family paraphrases via gpt-4o-mini, A1 synonyms BANNED."""
    try:
        from tsqa.llm.factory import create_llm_client
        client = create_llm_client(provider="openai")
    except Exception as e:
        print(f"[warn] paraphrase layer skipped (no OpenAI client: {e})")
        return []
    sys_p = ("You rewrite a request for a single statistic of a list of numbers as a "
             "natural question a person might ask. Keep it asking for the SAME single "
             "statistic. Output ONLY the question, one line.")
    rows = []
    targets = [q for q in _BANNED]
    for ts in series:
        q = rng.choice(targets)
        banned = ", ".join(_BANNED[q])
        usr = (f"Original: 'What is the {_QPHRASE[q]} of the series?'\n"
               f"Rewrite it WITHOUT using any of these words: {banned}. "
               f"Use everyday wording. One question only.")
        try:
            para = str(client.generate(sys_p, usr)).strip().splitlines()[0].strip()
        except Exception as e:
            print(f"[warn] paraphrase gen failed: {e}")
            continue
        if not para or any(b in para.lower() for b in _BANNED[q]):
            continue  # the generator leaked a banned word → drop (keep the layer honest)
        g = _gold(_leaf(q), ts)
        if g is None:
            continue
        rows.append(("paraphrase", json.dumps(_leaf(q)), para, g, ts))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/numeric_stress.csv")
    ap.add_argument("--n-series", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-paraphrase", action="store_true",
                    help="skip the OpenAI paraphrase layer (offline/CI)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = load_base_series(args.n_series * 4, args.seed)
    if len(pool) < args.n_series:
        raise SystemExit(f"only {len(pool)} usable Base series found")

    def take(k):
        return [pool[i % len(pool)] for i in rng.sample(range(len(pool)), min(k, len(pool)))]

    rows = []
    rows += gen_clean(take(args.n_series), rng)
    rows += gen_compositional(take(args.n_series), rng)
    rows += gen_typo(take(args.n_series), rng)
    if not args.no_paraphrase:
        rows += gen_paraphrase(take(args.n_series), rng)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "layer", "intent_plan", "question", "ground_truth", "series"])
        for i, (layer, plan, q, g, ts) in enumerate(rows):
            w.writerow([f"stress_{i:04d}", layer, plan, q, g, json.dumps(ts)])

    from collections import Counter
    by = Counter(r[0] for r in rows)
    print(f"wrote {len(rows)} rows -> {out}")
    for layer, n in by.most_common():
        print(f"  {layer:14} {n}")


if __name__ == "__main__":
    main()
