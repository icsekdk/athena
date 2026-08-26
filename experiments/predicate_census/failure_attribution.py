"""
Failure attribution. For every scenario where the final status is a
PROVED_* that disagrees with the scenario's expected verdict, classify why
-- reusing the existing gold30_oracle_interface_<model>.json traces
produced by run_oracle_interface.py, no new LLM calls.

Three buckets, checked in this order:
  - SEARCH: unnecessary_questions_asked > 0 or soundness_failures_vs_own_state
    > 0 -- the proof-obligation search itself misbehaved. Expected count: 0.
  - FORMALIZATION_OR_LABEL: zero oracle calls were made at all (the proof
    closed on structural/given facts alone) yet the verdict still disagrees
    with the expected label -- the oracle was never in a position to cause
    this, so it isn't oracle failure; it's a mismatch between what the
    current formalization derives from the given facts and gold30's expected
    label.
  - ORACLE: at least one oracle call made, and at least one of those calls
    disagrees with independent ground truth -- the search asked the right
    question(s) but the answer(s) were wrong.

UNRESOLVED is never a failure by this project's convention (correct
behavior when the evidence genuinely doesn't establish a verdict) and is
excluded from the classification entirely.

Usage: python3 failure_attribution.py --model gpt_5_5
       python3 failure_attribution.py --model all
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS = ["gemma3_4b", "llama3_1_8b", "claude_sonnet_5", "gpt_5_5"]


def classify(r):
    if r["unnecessary_questions_asked"] > 0 or len(r["soundness_failures_vs_own_state"]) > 0:
        return "SEARCH"
    if not r["oracle_calls"]:
        return "FORMALIZATION_OR_LABEL"
    if any(not c["agrees"] for c in r["oracle_calls"]):
        return "ORACLE"
    # every oracle call made agrees with ground truth, search behaved, no
    # calls were skipped -- yet the verdict is still wrong. Not expected to
    # occur; surfaced explicitly rather than folded into another bucket.
    return "UNEXPLAINED"


def run_one(model_slug):
    path = ROOT / f"data/gold30_oracle_interface_{model_slug}.json"
    data = json.loads(path.read_text())

    misses = [r for r in data if r["final_status"].startswith("PROVED")
              and r["final_status"] != f"PROVED_{r['expected_verdict']}"]

    buckets = {"SEARCH": [], "FORMALIZATION_OR_LABEL": [], "ORACLE": [], "UNEXPLAINED": []}
    for r in misses:
        buckets[classify(r)].append(r["scenario"])

    n = len(data)
    n_unresolved = sum(1 for r in data if r["final_status"] == "UNRESOLVED")
    n_correct_proved = sum(1 for r in data if r["final_status"] == f"PROVED_{r['expected_verdict']}")

    print(f"\n=== {model_slug} ===")
    print(f"n={n}  PROVED-correct={n_correct_proved}  UNRESOLVED={n_unresolved}  PROVED-wrong={len(misses)}")
    for bucket, scenarios in buckets.items():
        if scenarios:
            print(f"  {bucket}: {len(scenarios)}  {scenarios}")
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    cli = ap.parse_args()
    targets = MODELS if cli.model == "all" else [cli.model]
    all_buckets = {}
    for m in targets:
        all_buckets[m] = run_one(m)

    print("\n=== SEARCH-failure invariant check ===")
    total_search = sum(len(b["SEARCH"]) for b in all_buckets.values())
    print(f"Total SEARCH-bucket misses across {len(targets)} model(s): {total_search}"
          + ("  <-- must be 0" if total_search else "  -- holds"))

    out_path = ROOT / "data/failure_attribution.json"
    out_path.write_text(json.dumps(all_buckets, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
