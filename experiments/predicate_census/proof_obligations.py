"""
Ground-truth proof-obligation computation, applied mechanically to all 30
gold30 scenarios.

For each scenario, U (the unknown-fact set, from gold30's independent ground
truth, data/gold30_independent_ground_truth.json) is small enough in every
observed case (2-8 facts) that full 2^|U| completion enumeration is directly
tractable -- so this script uses the literal exhaustive ground-truth
definition, not a monotone-bracket shortcut (which exists for scalability at
larger fact counts, not because the exhaustive definition is ever wrong).
This sidesteps needing to re-implement occurrence-polarity probing at all:
the exhaustive enumeration is correct unconditionally, including for
non-monotone predicates like is_required_by_law.

For each scenario, computes:
  - status: PROVED_PERMITTED / PROVED_DENIED / UNRESOLVED
  - critical set C: facts where some completion of the rest makes them
    decisive
  - minimum question count: exact decision-tree depth of the verdict function
    restricted to U, computed by brute-force minimax search over the already-built
    completion table (no extra souffle calls needed for this part)
  - for UNRESOLVED scenarios, a per-critical-fact consequence table

Usage: python3 proof_obligations.py
Writes: ../../data/gold30_proof_obligations.json
"""
import itertools
import json
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from decision_criticality import parse_fact, run_souffle

ROOT = Path(__file__).resolve().parents[2]


def completion_table(given, K, U, tmp, prefix):
    """Return {completion_tuple: verdict} for all 2^|U| completions of U."""
    table = {}
    names = [name for name, _args in U]
    for i, bits in enumerate(itertools.product([False, True], repeat=len(U))):
        asserted = [U[j] for j, b in enumerate(bits) if b]
        v = run_souffle(given + K + asserted, tmp / f"{prefix}_{i}")
        table[bits] = v
    return table, names


def min_decision_depth(table, n):
    """Exact minimax decision-tree depth: fewest adaptive TRUE/FALSE queries
    (over the n boolean vars) needed to determine the verdict in the worst case."""
    all_bits = list(table.keys())

    @lru_cache(maxsize=None)
    def depth(remaining_vars, fixed):
        # fixed: tuple of (var_index, bool) already decided
        fixed_d = dict(fixed)
        consistent = [b for b in all_bits if all(b[i] == v for i, v in fixed_d.items())]
        outcomes = {table[b] for b in consistent}
        if len(outcomes) <= 1:
            return 0
        best = None
        for v in remaining_vars:
            worst = 0
            for val in (False, True):
                nf = fixed + ((v, val),)
                nr = tuple(x for x in remaining_vars if x != v)
                d = depth(nr, nf)
                worst = max(worst, d)
            cand = 1 + worst
            if best is None or cand < best:
                best = cand
        return best

    return depth(tuple(range(n)), tuple())


def critical_set(table, names):
    n = len(names)
    critical = set()
    for i in range(n):
        for b in table:
            other = b[:i] + b[i + 1:]
            b0 = b[:i] + (False,) + b[i + 1:]
            b1 = b[:i] + (True,) + b[i + 1:]
            if table[b0] != table[b1]:
                critical.add(names[i])
                break
    return critical


def main():
    bench = json.loads((ROOT / "data/athena_bench_gold30.json").read_text())
    gt = json.loads((ROOT / "data/gold30_independent_ground_truth.json").read_text())
    gt_by_scenario = defaultdict(list)
    for r in gt:
        gt_by_scenario[r["scenario"]].append(r)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, sc in enumerate(bench["scenarios"]):
            sid = sc["id"].replace("athena-", "")
            expected = "PERMITTED" if sc["expected_verdict"] == "ALLOW" else "DENIED"

            facts = [parse_fact(f) for f in sc["facts"]]
            facts = [f for f in facts if f]
            given = [f for f in facts if f[0] in ("disclosure_attempted", "msg_contains")]

            rows = gt_by_scenario.get(sid, [])
            K = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                 for r in rows if r["independent_value"] == "TRUE"]
            U = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                 for r in rows if r["independent_value"] == "UNKNOWN"]

            table, names = completion_table(given, K, U, tmp, f"s{i}")
            outcomes = set(table.values())

            if len(outcomes) == 1:
                status = f"PROVED_{outcomes.pop()}"
                critical = []
                min_q = 0
            else:
                status = "UNRESOLVED"
                critical = sorted(critical_set(table, names))
                min_q = min_decision_depth(table, len(U))

            consequences = []
            if status == "UNRESOLVED":
                for name, args in U:
                    if name not in critical:
                        continue
                    idx = names.index(name)
                    v_true = {table[b] for b in table if b[idx]}
                    v_false = {table[b] for b in table if not b[idx]}
                    consequences.append({
                        "predicate": name, "args": args,
                        "if_true_possible": sorted(v_true),
                        "if_false_possible": sorted(v_false),
                    })

            results.append({
                "scenario": sid,
                "expected_verdict": expected,
                "n_known_true": len(K),
                "n_unknown": len(U),
                "status": status,
                "matches_expected": (status == f"PROVED_{expected}") or
                                     (status == "UNRESOLVED"),
                "critical_facts": critical,
                "n_critical": len(critical),
                "min_questions": min_q,
                "consequence_table": consequences,
            })
            print(f"{sid}: expected={expected} status={status} "
                  f"n_unknown={len(U)} n_critical={len(critical)} min_q={min_q}")

    (ROOT / "data/gold30_proof_obligations.json").write_text(json.dumps(results, indent=2))

    n = len(results)
    proved_correct = sum(1 for r in results if r["status"] == f"PROVED_{r['expected_verdict']}")
    proved_wrong = sum(1 for r in results if r["status"].startswith("PROVED_")
                        and r["status"] != f"PROVED_{r['expected_verdict']}")
    unresolved = sum(1 for r in results if r["status"] == "UNRESOLVED")
    total_min_q = sum(r["min_questions"] for r in results)

    print(f"\n=== SUMMARY (n={n}) ===")
    print(f"PROVED and matches expected: {proved_correct}/{n}")
    print(f"PROVED but WRONG verdict (formalization/grounding disagrees with corpus): {proved_wrong}/{n}")
    print(f"UNRESOLVED (genuinely contingent on remaining unknowns): {unresolved}/{n}")
    print(f"Total minimum questions across all 30 scenarios: {total_min_q}")


if __name__ == "__main__":
    main()
