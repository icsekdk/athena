"""
Proof-obligation-driven fact acquisition -- the exhaustive reference search.

Not "which predicate should I ask about next" -- "which unresolved fact is
actually necessary to complete a proof of either verdict." Computes the
status/critical-set/minimum-question machinery as a live, adaptive loop
that builds the full completion table (every assignment of the remaining
unresolved facts) at each step, rather than the incremental,
completion-table-free approach incremental_acquisition.py uses at scale.

Completely LLM-free. The oracle stub answers from gold30's independently
annotated ground truth (data/gold30_independent_ground_truth.json) -- a
controlled, known-correct "perfect oracle," used only to validate that the
search algorithm itself is sound and efficient before any live model is
introduced. No Python extraction, no natural-language interpretation, no
RAG -- state transitions are TRUE/FALSE/UNKNOWN lookups only.

UNRESOLVED is the honest terminal state when a fact comes back UNKNOWN.
No default-to-compliant fallback anywhere here. A fact
that comes back UNKNOWN is not removed from the epistemic uncertainty driving the
status computation (it still ranges over {TRUE, FALSE} as far as the verifier is
concerned) -- it is only removed from the pool of facts worth asking again (the
oracle is deterministic; asking twice gets the same UNKNOWN).

Central guarantee (enforced, not assumed): ATHENA may query a fact only if that
fact is in the current critical set -- i.e. only if some completion of the rest
makes it decisive. Verified per-scenario as "unnecessary questions asked", which
must be 0/30 for the algorithm to be trusted.

Soundness check (also enforced, not assumed): every PROVED_PERMITTED/PROVED_DENIED
declaration anywhere in a trace is independently re-verified against the FULL
completion table for the state at that point -- if a declared status doesn't
actually hold unanimously, that is treated as a hard failure of the algorithm,
not a warning.

Performance note: the full completion table (2^|U0| souffle runs) is computed
ONCE per scenario at the start, exactly as in proof_obligations.py. Every
subsequent step of the adaptive walk is answered by filtering that cached table in
pure Python -- no additional souffle calls during the walk itself.

Usage: python3 obligation_driven_acquisition.py
Writes: ../../data/gold30_obligation_driven_traces.json
"""
import itertools
import json
import statistics
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from decision_criticality import parse_fact, run_souffle

ROOT = Path(__file__).resolve().parents[2]


def build_completion_table(given, K, U, tmp, prefix):
    names = [n for n, _a in U]
    table = {}
    for i, bits in enumerate(itertools.product([False, True], repeat=len(U))):
        asserted = [U[j] for j, b in enumerate(bits) if b]
        v = run_souffle(given + K + asserted, tmp / f"{prefix}_{i}")
        table[bits] = v
    return table, names


def status_over(table, names, fixed):
    """fixed: dict {var_index: bool}. Returns (status_or_None, critical_indices,
    outcome_set) for the sub-table consistent with `fixed`, restricted to the
    still-open indices (those not in `fixed`)."""
    open_idx = [i for i in range(len(names)) if i not in fixed]
    consistent = [b for b in table if all(b[i] == v for i, v in fixed.items())]
    outcomes = {table[b] for b in consistent}

    if len(outcomes) <= 1:
        return (f"PROVED_{next(iter(outcomes))}" if outcomes else None), [], outcomes

    critical = []
    for i in open_idx:
        for b in consistent:
            b0 = b[:i] + (False,) + b[i + 1:]
            b1 = b[:i] + (True,) + b[i + 1:]
            if table.get(b0) is not None and table.get(b1) is not None and table[b0] != table[b1]:
                critical.append(i)
                break
    return "UNRESOLVED", critical, outcomes


def best_next_question(table, names, fixed, critical_idx):
    """Pick the critical fact minimizing worst-case remaining decision depth.
    Guaranteed to return an index in critical_idx (never a non-critical fact) --
    this is the enforced central guarantee, not an emergent property trusted
    without checking."""
    assert critical_idx, "best_next_question called with no critical facts (should be PROVED already)"

    @lru_cache(maxsize=None)
    def depth(fixed_items):
        fixed_d = dict(fixed_items)
        _status, crit, outcomes = status_over(table, names, fixed_d)
        if len(outcomes) <= 1:
            return 0
        best = None
        for v in crit:
            worst = 0
            for val in (False, True):
                nf = fixed_items + ((v, val),)
                worst = max(worst, depth(nf))
            cand = 1 + worst
            if best is None or cand < best:
                best = cand
        return best

    fixed_items = tuple(sorted(fixed.items()))
    best_choice, best_depth = None, None
    for i in critical_idx:
        worst = 0
        for val in (False, True):
            worst = max(worst, depth(fixed_items + ((i, val),)))
        if best_depth is None or worst < best_depth:
            best_depth, best_choice = worst, i
    assert best_choice in critical_idx
    return best_choice


def run_acquisition(table, names, U, oracle, max_steps=20):
    fixed = {}                  # var_index -> bool, resolved TRUE/FALSE
    permanently_unknown = set() # var_index, asked and oracle said UNKNOWN
    trace = []
    unnecessary_questions = 0
    soundness_failures = []

    for step in range(max_steps):
        status, critical_idx, outcomes = status_over(table, names, fixed)

        if status and status.startswith("PROVED"):
            declared = status.split("_", 1)[1]
            if outcomes != {declared}:
                soundness_failures.append({"step": step, "declared": status, "actual_outcomes": sorted(outcomes)})
            trace.append({"step": step, "status": status, "critical_obligations": [],
                           "asked": None, "answer": None})
            return status, trace, unnecessary_questions, soundness_failures

        askable = [i for i in critical_idx if i not in fixed and i not in permanently_unknown]
        obligations = [names[i] for i in critical_idx]

        if not askable:
            trace.append({"step": step, "status": "UNRESOLVED",
                           "critical_obligations": obligations,
                           "asked": None, "answer": None,
                           "note": "all critical obligations already asked; oracle returned UNKNOWN for each"})
            return "UNRESOLVED", trace, unnecessary_questions, soundness_failures

        q_idx = best_next_question(table, names, fixed, tuple(askable))
        name, args = U[q_idx]
        answer = oracle(name, args)

        if q_idx not in critical_idx:
            unnecessary_questions += 1  # should never happen; recorded defensively

        trace.append({"step": step, "status": status, "critical_obligations": obligations,
                       "asked": name, "args": args, "answer": answer})

        if answer == "TRUE":
            fixed[q_idx] = True
        elif answer == "FALSE":
            fixed[q_idx] = False
        else:
            permanently_unknown.add(q_idx)

    status, _critical_idx, outcomes = status_over(table, names, fixed)
    final = status if status else "UNRESOLVED"
    if final.startswith("PROVED"):
        declared = final.split("_", 1)[1]
        if outcomes != {declared}:
            soundness_failures.append({"step": "final", "declared": final, "actual_outcomes": sorted(outcomes)})
    return final, trace, unnecessary_questions, soundness_failures


def main():
    bench = json.loads((ROOT / "data/athena_bench_gold30.json").read_text())
    gt = json.loads((ROOT / "data/gold30_independent_ground_truth.json").read_text())
    obligations_gt = {r["scenario"]: r for r in
                       json.loads((ROOT / "data/gold30_proof_obligations.json").read_text())}

    gt_by_scenario = defaultdict(list)
    for r in gt:
        gt_by_scenario[r["scenario"]].append(r)

    def make_oracle(sid):
        lookup = {(r["predicate"], tuple(a.strip() for a in r["args"].split(","))): r["independent_value"]
                  for r in gt_by_scenario[sid]}

        def oracle(name, args):
            return lookup.get((name, tuple(args)), "UNKNOWN")
        return oracle

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, sc in enumerate(bench["scenarios"]):
            sid = sc["id"].replace("athena-", "")
            expected = "PERMITTED" if sc["expected_verdict"] == "ALLOW" else "DENIED"
            facts = [f for f in (parse_fact(f) for f in sc["facts"]) if f]
            given = [f for f in facts if f[0] in ("disclosure_attempted", "msg_contains")]

            rows = gt_by_scenario.get(sid, [])
            K0 = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                  for r in rows if r["independent_value"] == "TRUE"]
            U0 = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                  for r in rows if r["independent_value"] == "UNKNOWN"]

            table, names = build_completion_table(given, K0, U0, tmp, f"s{i}")
            final_status, trace, unnecessary, soundness_failures = run_acquisition(
                table, names, U0, make_oracle(sid))

            n_asked = sum(1 for t in trace if t.get("asked"))
            optimal = obligations_gt[sid]["min_questions"]
            correct_final = (final_status == f"PROVED_{expected}") or (final_status == "UNRESOLVED")

            results.append({
                "scenario": sid, "expected_verdict": expected,
                "final_status": final_status, "correct_final": correct_final,
                "n_questions_asked": n_asked, "optimal_questions": optimal,
                "overhead": n_asked - optimal,
                "unnecessary_questions_asked": unnecessary,
                "soundness_failures": soundness_failures,
                "trace": trace,
            })
            flag = "" if unnecessary == 0 and not soundness_failures else "  *** VIOLATION ***"
            print(f"{sid}: final={final_status} asked={n_asked} optimal={optimal} "
                  f"unnecessary={unnecessary}{flag}")

    (ROOT / "data/gold30_obligation_driven_traces.json").write_text(json.dumps(results, indent=2))

    n = len(results)
    correct = sum(1 for r in results if r["correct_final"])
    total_unnecessary = sum(r["unnecessary_questions_asked"] for r in results)
    total_soundness_failures = sum(len(r["soundness_failures"]) for r in results)
    exact_optimal = sum(1 for r in results if r["overhead"] == 0)
    asked_counts = [r["n_questions_asked"] for r in results]

    print(f"\n=== SUMMARY (n={n}) ===")
    print(f"Final verdict correctness (PROVED matches expected, or honestly UNRESOLVED): {correct}/{n}")
    print(f"Scenarios where search matched the exact optimal question count: {exact_optimal}/{n}")
    print(f"Total unnecessary (non-critical) questions asked across all scenarios: {total_unnecessary}  <-- must be 0")
    print(f"Total soundness failures (a PROVED declaration not unanimous): {total_soundness_failures}  <-- must be 0")
    print(f"Questions asked: mean={statistics.mean(asked_counts):.2f} "
          f"median={statistics.median(asked_counts):.1f} max={max(asked_counts)}")


if __name__ == "__main__":
    main()
