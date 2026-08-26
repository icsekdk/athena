"""
Refined one-shot extraction -- a fourth baseline arm alongside Direct LLM /
ETV-closed / ETV-open / ATHENA: extract all candidate facts once (Baseline
B's first pass, already computed and stored in
data/baseline_comparison_<corpus>_<model>.json's "b_detail"), then give the
model ONE additional call that shows it the narrative plus its own first-pass
answers and asks it to review and correct them, before the (refined) fact
set is handed to the same Datalog verifier used by every other baseline.

This reuses the existing baseline-comparison run's first-pass answers rather
than re-querying the model for them -- only the refinement call is new.
Requires data/baseline_comparison_<corpus>_<model>.json to already exist for
the requested model/corpus.

Usage: python3 refined_baseline.py --model claude-sonnet-5 --provider anthropic --corpus data/athena_bench_tier2_132.json
Writes: ../../data/refined_baseline_<corpus>_<model>.json
"""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import run_baselines as base  # noqa: E402
from baseline_comparison import build_b_scope  # noqa: E402
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]


def refinement_prompt(narrative, given, candidates, first_pass):
    """first_pass: dict {1-based index: TRUE/FALSE/UNKNOWN} from the stored
    b_detail of the same scenario's first extraction pass."""
    given_txt = "\n".join(base.fmt_fact(n, a) for n, a in given)
    lines = []
    for i, (name, args_) in enumerate(candidates, start=1):
        prev = first_pass.get(i, "UNKNOWN")
        lines.append(f"{i}. {base.fmt_fact(name, args_)}  [your previous answer: {prev}]")
    q_txt = "\n".join(lines)
    return f"""You previously extracted candidate facts from the narrative below to feed a formal legal verifier, producing the answers shown next to each claim. You now have one opportunity to REVIEW and CORRECT those answers -- re-read the narrative carefully and check each claim again. Fix any answer you got wrong: evidence you missed the first time, a claim you marked UNKNOWN that the narrative actually establishes (or the reverse), or a claim you misread.

NARRATIVE:
{narrative}

CONTEXT (already established, given to you, do not re-answer):
{given_txt}

For EACH numbered claim below, give your FINAL answer after re-checking: TRUE, FALSE, or UNKNOWN. TRUE = narrative directly states or clearly implies it. FALSE = narrative directly states or clearly implies the opposite. UNKNOWN = narrative does not address it, or is too ambiguous. You may keep your previous answer if you still believe it is correct.

{q_txt}

Respond with exactly one line per claim, in this format, nothing else:
<number>: <TRUE|FALSE|UNKNOWN>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True, choices=["ollama", "anthropic", "openai"])
    ap.add_argument("--corpus", default="data/athena_bench_tier2_132.json")
    cli = ap.parse_args()
    base.MODEL, base.PROVIDER = cli.model, cli.provider
    model_slug = re.sub(r'[^a-z0-9]+', '_', cli.model.lower())
    is_gold30 = cli.corpus == "data/athena_bench_gold30.json"
    corpus_slug = "gold30" if is_gold30 else Path(cli.corpus).stem.replace("athena_bench_", "")

    bench = json.loads((ROOT / cli.corpus).read_text())
    gt_by_scenario = defaultdict(list)
    if is_gold30:
        gt = json.loads((ROOT / "data/gold30_independent_ground_truth.json").read_text())
        for r in gt:
            gt_by_scenario[r["scenario"]].append(r)

    baseline_path = ROOT / f"data/baseline_comparison_{corpus_slug}_{model_slug}.json"
    if not baseline_path.exists():
        raise SystemExit(f"Missing prerequisite {baseline_path} -- run baseline_comparison.py first.")
    baseline = {r["scenario"]: r for r in json.loads(baseline_path.read_text())}

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, sc in enumerate(bench["scenarios"]):
            sid = sc["id"].replace("athena-", "")
            expected = "PERMITTED" if sc["expected_verdict"] == "ALLOW" else "DENIED"
            narrative_text = sc["narrative"]["situation"] + "\n\nQuestion: " + sc["narrative"]["question"]

            given, candidates = build_b_scope(sc, gt_by_scenario)
            prior = baseline.get(sid)
            if prior is None:
                print(f"{sid}: SKIPPED (not in baseline_comparison output)")
                continue

            if not candidates:
                # Same as baseline_b's own zero-candidate shortcut -- nothing
                # to refine, closed-world verdict is already final.
                refined_verdict = prior["baseline_b_closed"]
                refined_detail = {}
            else:
                # Reconstruct the first pass's 1-indexed TRUE/FALSE/UNKNOWN
                # from the stored b_detail (keyed by rendered fact string,
                # in the same order build_b_scope reproduces deterministically).
                first_pass = {}
                for idx, (name, args_) in enumerate(candidates, start=1):
                    key = f"{name}(" + ",".join(args_) + ")"
                    val = prior["b_detail"].get(key)
                    if val is None:
                        # fmt_fact join style differs (", " vs ","); try that too
                        key2 = f"{name}(" + ", ".join(args_) + ")"
                        val = prior["b_detail"].get(key2, "UNKNOWN")
                    first_pass[idx] = val

                prompt = refinement_prompt(narrative_text, given, candidates, first_pass)
                resp = base.ollama(prompt)
                parsed = base.parse_extraction(resp, len(candidates))

                true_facts = list(given)
                refined_detail = {}
                for idx, (name, args_) in enumerate(candidates, start=1):
                    val = parsed.get(idx, first_pass.get(idx, "UNKNOWN"))  # no answer -> keep first pass, not blank UNKNOWN
                    refined_detail[f"{name}({','.join(args_)})"] = val
                    if val == "TRUE":
                        true_facts.append((name, args_))

                n_allowed, n_denied = base.run_souffle(true_facts, tmp / f"s{i}")
                refined_verdict = "PERMITTED" if n_allowed else "DENIED"

            n_changed = sum(1 for k, v in refined_detail.items()
                             if v != prior["b_detail"].get(k, "UNKNOWN")) if refined_detail else 0

            results.append({
                "scenario": sid, "expected": expected,
                "baseline_a": prior["baseline_a"],
                "baseline_b_closed": prior["baseline_b_closed"],
                "baseline_b_open": prior["baseline_b_open"],
                "refined_verdict": refined_verdict,
                "n_candidates": prior["b_n_candidates"],
                "n_answers_changed_by_refinement": n_changed,
                "refined_detail": refined_detail,
                "athena_verdict": prior["athena_verdict"],
                "athena_n_questions": prior["athena_n_questions"],
            })
            print(f"{sid}: expected={expected} B-closed={prior['baseline_b_closed']} "
                  f"Refined={refined_verdict} (changed={n_changed}/{prior['b_n_candidates']}) "
                  f"ATHENA={prior['athena_verdict']}")

    out_path = ROOT / f"data/refined_baseline_{corpus_slug}_{model_slug}.json"
    out_path.write_text(json.dumps(results, indent=2))

    def score(key):
        n = len(results)
        correct = sum(1 for r in results if r[key] == r["expected"])
        false_permit = sum(1 for r in results if r["expected"] == "DENIED" and r[key] == "PERMITTED")
        false_deny = sum(1 for r in results if r["expected"] == "PERMITTED" and r[key] == "DENIED")
        unresolved = sum(1 for r in results if r[key] == "UNRESOLVED")
        return correct, false_permit, false_deny, unresolved, n

    print(f"\n=== SUMMARY ({cli.model}, {corpus_slug}, n={len(results)}) ===")
    for key in ["baseline_a", "baseline_b_closed", "refined_verdict", "athena_verdict"]:
        c, fp, fd, unr, n = score(key)
        print(f"{key}: correct={c}/{n} false_PERMIT={fp} false_DENY={fd} UNRESOLVED={unr}")
    total_changed = sum(r["n_answers_changed_by_refinement"] for r in results)
    total_candidates = sum(r["n_candidates"] for r in results)
    print(f"Refinement changed {total_changed}/{total_candidates} first-pass answers")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
