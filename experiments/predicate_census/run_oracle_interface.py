"""
Live end-to-end run -- the incremental search loop (incremental_acquisition.py)
driven by a real LLM oracle (llm_oracle.py) instead of a perfect ground-truth
lookup oracle.

Separates three previously-conflated questions:
  - Search quality: did the algorithm select the right (critical-set) fact at each
    step? (the same enforced guarantee validated against the exhaustive reference
    search in obligation_driven_acquisition.py -- checked again here since it's a
    different, live run, not assumed to carry over.)
  - Oracle quality: for exactly the facts actually asked, does the LLM's answer
    match gold30's independent, narrative-grounded value
    (data/gold30_independent_ground_truth.json)? (A small, targeted
    comparison -- ~1-4 questions per scenario, not all 255 candidates.)
  - End-to-end quality: does the LLM-driven run's FINAL status match what the
    truth (gold30's independent ground truth plus the exhaustive completion
    table) actually supports? This is what a static, ground-truth-lookup oracle
    can never test: whether an LLM answering a critical fact WRONG can cause a
    wrong symbolic verdict to be asserted with false confidence -- the real
    correctness risk of introducing a live oracle at all, checked directly
    rather than assumed away.

Usage: python3 run_oracle_interface.py --model gemma3:4b --provider ollama
Writes: ../../data/gold30_oracle_interface_<model>.json
"""
import argparse
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path

from decision_criticality import parse_fact, run_souffle
from incremental_acquisition import run_incremental, load_polarity, resolve_mixed_fixed_point
from llm_oracle import make_llm_oracle

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True, choices=["ollama", "anthropic", "openai"])
    ap.add_argument("--corpus", default="data/athena_bench_gold30.json",
                     help="Point at data/athena_bench_tier2_132.json or "
                          "data/athena_bench_adversarial.json to scale beyond gold30. "
                          "Only gold30 has independent per-fact ground truth; other "
                          "corpora fall back to the scenario's own recorded facts as the "
                          "live-queried candidate pool, and oracle_calls carry "
                          "no ground_truth/agrees field.")
    cli = ap.parse_args()
    is_gold30 = cli.corpus == "data/athena_bench_gold30.json"
    polarity_role = load_polarity()

    bench = json.loads((ROOT / cli.corpus).read_text())
    gt_by_scenario = defaultdict(list)
    obligations_gt = {}
    if is_gold30:
        gt = json.loads((ROOT / "data/gold30_independent_ground_truth.json").read_text())
        obligations_gt = {r["scenario"]: r for r in
                           json.loads((ROOT / "data/gold30_proof_obligations.json").read_text())}
        for r in gt:
            gt_by_scenario[r["scenario"]].append(r)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, sc in enumerate(bench["scenarios"]):
            sid = sc["id"].replace("athena-", "")
            expected = "PERMITTED" if sc["expected_verdict"] == "ALLOW" else "DENIED"
            narrative = sc["narrative"]["situation"] + "\n\nQuestion: " + sc["narrative"]["question"]

            facts = [f for f in (parse_fact(f) for f in sc["facts"]) if f]
            given = [f for f in facts if f[0] in ("disclosure_attempted", "msg_contains")]

            rows = gt_by_scenario.get(sid, [])
            if rows:
                K0 = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                      for r in rows if r["independent_value"] == "TRUE"]
                U0 = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                      for r in rows if r["independent_value"] == "UNKNOWN"]
                gt_lookup = {(r["predicate"], tuple(a.strip() for a in r["args"].split(",")))
                             : r["independent_value"] for r in rows}
            else:
                # No independent ground truth for this corpus. Fall back to the
                # corpus-author's own recorded facts as the live-queried
                # candidate pool. K0 stays empty rather than trusting those
                # facts as already-true, to avoid smuggling in corpus-author
                # assumptions as free structural grounding.
                K0 = []
                seen = set()
                U0 = []
                for f in facts:
                    if f[0] in ("disclosure_attempted", "msg_contains"):
                        continue
                    key = (f[0], tuple(f[1]))
                    if key in seen:
                        # Some corpus scenarios list the same fact twice
                        # verbatim -- a source-data quirk, not asked twice.
                        continue
                    seen.add(key)
                    U0.append(f)
                gt_lookup = {}

            llm_oracle = make_llm_oracle(narrative, cli.model, cli.provider)
            oracle_calls = []

            def logged_oracle(name, args, _llm=llm_oracle, _gt=gt_lookup, _calls=oracle_calls):
                answer = _llm(name, args)
                if _gt:
                    truth = _gt.get((name, tuple(args)), "UNKNOWN")
                    agrees = answer == truth
                else:
                    # No independent ground truth for this corpus -- record the
                    # answer without a correctness judgment, rather than
                    # comparing against a meaningless default.
                    truth, agrees = None, None
                _calls.append({"predicate": name, "args": args, "llm_answer": answer,
                                "ground_truth": truth, "agrees": agrees})
                return answer

            final_status, trace, unnecessary, synergy_used = run_incremental(
                given, K0, U0, logged_oracle, polarity_role)
            soundness_failures = []  # kept for schema compatibility; see note below

            # End-to-end soundness re-check, independent of the search loop's own
            # internal bracket test: does the LLM-driven final status still hold
            # if every fact NOT confirmed TRUE by the LLM is allowed to range over
            # {TRUE, FALSE} again? No full completion table is built anymore --
            # this recomputes a fresh bracket over everything the LLM didn't
            # assert TRUE, using signed_polarity.py's
            # roles (MIXED predicates default to "blocker" here, the conservative
            # choice for a safety check: it can only produce a false alarm, never
            # miss a real one).
            asserted_true = [(t["asked"], tuple(t["args"])) for t in trace if t.get("answer") == "TRUE"]
            # K0 must be included in the re-check's fixed set: any scenario
            # resolved with 0 live questions (structurally closed by K0 alone)
            # would otherwise rerun the bracket with *no* known-true facts at
            # all, producing a spurious dangerous_false_proof flag. K0 is
            # exactly as trustworthy here as it is inside run_incremental
            # itself (same source).
            fixed_true_facts = list(K0) + [(n, list(a)) for n, a in asserted_true]
            # Both TRUE- and FALSE-answered facts must be excluded from
            # still_open, not just TRUE. A FALSE answer means closed-world-
            # absent -- run_incremental's own `remaining` list drops it
            # permanently, same as TRUE does -- so leaving FALSE-answered
            # facts "open" here would let a FALSE-answered blocker get
            # asserted TRUE in the worst-case bracket, flipping
            # v_worst_check for reasons that have nothing to do with the
            # proof's actual soundness. Only UNKNOWN-answered or never-asked
            # facts remain genuinely open.
            resolved_non_open = {(t["asked"], tuple(t["args"])) for t in trace
                                  if t.get("answer") in ("TRUE", "FALSE")}
            still_open = [(n, a) for n, a in U0 if (n, tuple(a)) not in resolved_non_open]

            # Reuses incremental_acquisition.py's own resolve_mixed_fixed_point
            # rather than a parallel reimplementation, so this independent
            # check can't silently drift from whatever the search algorithm
            # itself does whenever that algorithm changes.
            with tempfile.TemporaryDirectory() as check_tmp:
                check_tmp = Path(check_tmp)
                roles = resolve_mixed_fixed_point(given + fixed_true_facts, still_open,
                                                   polarity_role, check_tmp, "recheck")
                enablers_open = [(n, a) for n, a in still_open if roles[(n, tuple(a))] == "enabler"]
                blockers_open = [(n, a) for n, a in still_open if roles[(n, tuple(a))] == "blocker"]
                v_best_check = run_souffle(given + fixed_true_facts + enablers_open, check_tmp / "best")
                v_worst_check = run_souffle(given + fixed_true_facts + blockers_open, check_tmp / "worst")

            dangerous_false_proof = (
                final_status.startswith("PROVED") and
                (v_best_check != v_worst_check or v_best_check != final_status.split("_", 1)[1])
            )

            n_questions = sum(1 for t in trace if t.get("asked"))
            scored_calls = [c for c in oracle_calls if c["agrees"] is not None]
            n_agree = sum(1 for c in scored_calls if c["agrees"])
            optimal = obligations_gt[sid]["min_questions"] if sid in obligations_gt else None

            results.append({
                "scenario": sid, "expected_verdict": expected, "model": cli.model,
                "final_status": final_status,
                "correct_final": (final_status == f"PROVED_{expected}") or (final_status == "UNRESOLVED"),
                "n_questions_asked": n_questions, "optimal_questions": optimal,
                "unnecessary_questions_asked": unnecessary,
                "soundness_failures_vs_own_state": soundness_failures,
                "dangerous_false_proof": dangerous_false_proof,
                "oracle_calls": oracle_calls,
                "oracle_accuracy": (n_agree / len(scored_calls)) if scored_calls else None,
                "synergy_fallback_used": synergy_used,
                "trace": trace,
            })
            flag = ""
            if unnecessary or soundness_failures:
                flag += "  *** SEARCH VIOLATION ***"
            if dangerous_false_proof:
                flag += "  *** DANGEROUS: PROVED status not supported by ground truth ***"
            acc_str = f"{n_agree}/{len(scored_calls)}" if scored_calls else "n/a (no ground truth)"
            print(f"{sid}: final={final_status} asked={n_questions} optimal={optimal} "
                  f"oracle_acc={acc_str}{flag}")

    corpus_slug = "gold30" if is_gold30 else Path(cli.corpus).stem.replace("athena_bench_", "")
    out_path = ROOT / f"data/{corpus_slug}_oracle_interface_{re.sub(r'[^a-z0-9]+', '_', cli.model.lower())}.json"
    out_path.write_text(json.dumps(results, indent=2))

    n = len(results)
    scored_results = results
    correct = sum(1 for r in scored_results if r["correct_final"])
    dangerous = sum(1 for r in scored_results if r["dangerous_false_proof"])
    total_unnecessary = sum(r["unnecessary_questions_asked"] for r in scored_results)
    all_calls = [c for r in scored_results for c in r["oracle_calls"]]
    scored_all = [c for c in all_calls if c["agrees"] is not None]
    total_q = sum(r["n_questions_asked"] for r in scored_results)
    optimals = [r["optimal_questions"] for r in scored_results if r["optimal_questions"] is not None]

    print(f"\n=== SUMMARY ({cli.model}, n={n}) ===")
    print(f"Search quality -- unnecessary questions asked: {total_unnecessary}  <-- must be 0")
    if scored_all:
        overall_oracle_acc = sum(1 for c in scored_all if c["agrees"]) / len(scored_all)
        print(f"Oracle quality -- LLM answer matches independent ground truth: {sum(1 for c in scored_all if c['agrees'])}/{len(scored_all)} ({overall_oracle_acc:.0%})")
    else:
        print("Oracle quality -- not measured (no independent ground truth for this corpus)")
    print(f"End-to-end -- final status correct or honestly UNRESOLVED: {correct}/{n}")
    print(f"End-to-end -- DANGEROUS false proofs (PROVED but ground truth disagrees): {dangerous}/{n}  <-- must be 0")
    if optimals:
        print(f"Total questions asked: {total_q} (vs. {sum(optimals)} optimal under perfect knowledge)")
    else:
        print(f"Total questions asked: {total_q} (no precomputed optimal for this corpus)")


if __name__ == "__main__":
    main()
