"""
Frozen baseline comparison -- Baseline A (direct LLM verdict, no
verifier), Baseline B (batch fact-extraction, then one verifier pass), and
ATHENA (the frozen iterative loop), all on gold30, same model
(gpt-5.5) for all three so the comparison isolates architecture, not model
choice.

Baseline A is reused verbatim from experiments/baselines/
run_baselines.py's baseline_a() -- narrative only, no predicates, one call.

Baseline B is a deliberate departure from that same script's original
hand-curated 18-predicate GUARDS candidate list, which predates this
project's full 172-predicate reachable-predicate census and includes a few
predicates later found to be internal-only, never LLM-facing. Instead this
reuses the *current* project's actual per-scenario candidate construction --
K0/U0 from gold30's independent ground truth
(data/gold30_independent_ground_truth.json), the same split
run_oracle_interface.py already uses to drive ATHENA's own search -- so the
comparison is batch-vs-iterative acquisition over the identical candidate
pool, not narrow-vs-full context.

ATHENA itself is not re-run here -- the frozen
data/gold30_oracle_interface_gpt_5_5.json is read directly.

Usage: python3 baseline_comparison.py --model gpt-5.5 --provider openai
Writes: ../../data/baseline_comparison_<corpus>_<model>.json
"""
import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import run_baselines as base  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def build_b_scope(sc, gt_by_scenario):
    """given = disclosure_attempted/msg_contains + K0 (09-confirmed TRUE);
    candidates = U0 (09-confirmed UNKNOWN) -- the same split
    run_oracle_interface.py uses to drive ATHENA's live search. Falls back
    to the corpus-author's own recorded facts (deduped -- some scenarios
    list the same fact twice verbatim) when no
    independent ground truth exists for this corpus, matching run_oracle_
    interface.py's --corpus fallback exactly."""
    facts = [f for f in (base.parse_fact(f) for f in sc["facts"]) if f]
    given = [f for f in facts if f[0] in ("disclosure_attempted", "msg_contains")]

    sid = sc["id"].replace("athena-", "")
    rows = gt_by_scenario.get(sid, [])
    if rows:
        K0 = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
              for r in rows if r["independent_value"] == "TRUE"]
        U0 = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
              for r in rows if r["independent_value"] == "UNKNOWN"]
        return given + K0, U0

    seen = set()
    U0 = []
    for f in facts:
        if f[0] in ("disclosure_attempted", "msg_contains"):
            continue
        key = (f[0], tuple(f[1]))
        if key in seen:
            continue
        seen.add(key)
        U0.append(f)
    return given, U0


def baseline_b(narrative_text, given, candidates, workdir):
    if not candidates:
        n_allowed, n_denied = base.run_souffle(given, workdir)
        closed = "PERMITTED" if n_allowed else "DENIED"
        return closed, closed, {}

    prompt = base.extraction_prompt(narrative_text, given, candidates)
    resp = base.ollama(prompt)
    parsed = base.parse_extraction(resp, len(candidates))

    true_facts = list(given)
    n_unknown = 0
    detail = {}
    for idx, (name, args_) in enumerate(candidates, start=1):
        val = parsed.get(idx, "UNKNOWN")
        detail[f"{name}({','.join(args_)})"] = val
        if val == "TRUE":
            true_facts.append((name, args_))
        elif val == "UNKNOWN":
            n_unknown += 1

    n_allowed, n_denied = base.run_souffle(true_facts, workdir)
    closed_verdict = "PERMITTED" if n_allowed else "DENIED"
    open_verdict = "UNRESOLVED" if n_unknown > 0 else closed_verdict
    return closed_verdict, open_verdict, detail


def main():
    global_ap = argparse.ArgumentParser()
    global_ap.add_argument("--model", required=True)
    global_ap.add_argument("--provider", required=True, choices=["ollama", "anthropic", "openai"])
    global_ap.add_argument("--corpus", default="data/athena_bench_gold30.json")
    cli = global_ap.parse_args()
    base.MODEL, base.PROVIDER = cli.model, cli.provider
    model_slug = re.sub(r'[^a-z0-9]+', '_', cli.model.lower())
    is_gold30 = cli.corpus == "data/athena_bench_gold30.json"
    corpus_slug = "gold30" if is_gold30 else Path(cli.corpus).stem.replace("athena_bench_", "")

    bench = json.loads((ROOT / cli.corpus).read_text())
    gt_by_scenario = defaultdict(list)
    obligations_gt = {}
    if is_gold30:
        gt = json.loads((ROOT / "data/gold30_independent_ground_truth.json").read_text())
        obligations_gt = {r["scenario"]: r for r in
                           json.loads((ROOT / "data/gold30_proof_obligations.json").read_text())}
        for r in gt:
            gt_by_scenario[r["scenario"]].append(r)

    athena = {r["scenario"]: r for r in
              json.loads((ROOT / f"data/{corpus_slug}_oracle_interface_{model_slug}.json").read_text())}

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, sc in enumerate(bench["scenarios"]):
            sid = sc["id"].replace("athena-", "")
            expected = "PERMITTED" if sc["expected_verdict"] == "ALLOW" else "DENIED"
            narrative_text = sc["narrative"]["situation"] + "\n\nQuestion: " + sc["narrative"]["question"]

            a_verdict, a_raw = base.baseline_a(narrative_text)

            given, candidates = build_b_scope(sc, gt_by_scenario)
            b_closed, b_open, b_detail = baseline_b(narrative_text, given, candidates, tmp / f"s{i}")

            a_row = athena[sid]
            athena_verdict = ("UNRESOLVED" if a_row["final_status"] == "UNRESOLVED"
                               else a_row["final_status"].split("_", 1)[1])

            if sid in obligations_gt:
                optimal = obligations_gt[sid]["min_questions"]
            else:
                # No exhaustive optimal exists for this corpus (23B) -- use
                # what ATHENA's own live adaptive search actually needed as
                # the comparison point instead of a theoretical minimum.
                optimal = a_row["n_questions_asked"]
            b_unnecessary = max(0, len(candidates) - optimal)

            results.append({
                "scenario": sid, "expected": expected,
                "baseline_a": a_verdict,
                "baseline_b_closed": b_closed, "baseline_b_open": b_open,
                "b_n_candidates": len(candidates), "b_unnecessary": b_unnecessary,
                "b_detail": b_detail,
                "athena_verdict": athena_verdict,
                "athena_n_questions": a_row["n_questions_asked"],
                "athena_unnecessary": a_row["unnecessary_questions_asked"],
            })
            print(f"{sid}: expected={expected} A={a_verdict} B-closed={b_closed} B-open={b_open} "
                  f"(|U0|={len(candidates)}, optimal={optimal}, unnecessary={b_unnecessary}) "
                  f"ATHENA={athena_verdict} (asked={a_row['n_questions_asked']})")

    out_path = ROOT / f"data/baseline_comparison_{corpus_slug}_{model_slug}.json"
    out_path.write_text(json.dumps(results, indent=2))

    def score(key):
        n = len(results)
        correct = sum(1 for r in results if r[key] == r["expected"])
        false_permit = sum(1 for r in results if r["expected"] == "DENIED" and r[key] == "PERMITTED")
        false_deny = sum(1 for r in results if r["expected"] == "PERMITTED" and r[key] == "DENIED")
        unresolved = sum(1 for r in results if r[key] == "UNRESOLVED")
        return correct, false_permit, false_deny, unresolved, n

    print(f"\n=== SUMMARY ({cli.model}, n={len(results)}) ===")
    for key, calls in [("baseline_a", 1), ("baseline_b_closed", 1), ("baseline_b_open", 1), ("athena_verdict", None)]:
        c, fp, fd, unr, n = score(key)
        print(f"{key}: correct={c}/{n} false_PERMIT={fp} false_DENY={fd} UNRESOLVED={unr}")

    total_q = sum(r["athena_n_questions"] for r in results)
    if is_gold30:
        total_optimal = sum(obligations_gt[r["scenario"]]["min_questions"] for r in results)
        optimal_label = "optimal"
    else:
        total_optimal = total_q  # no exhaustive optimal for this corpus; ATHENA's own count stands in
        optimal_label = "ATHENA's own count (no exhaustive optimal for this corpus)"
    total_b_candidates = sum(r["b_n_candidates"] for r in results)
    total_b_unnecessary = sum(r["b_unnecessary"] for r in results)
    print(f"ATHENA questions: {total_q} ({optimal_label} {total_optimal}), unnecessary=0")
    print(f"Baseline B candidates extracted: {total_b_candidates}, unnecessary={total_b_unnecessary} "
          f"({optimal_label} {total_optimal})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
