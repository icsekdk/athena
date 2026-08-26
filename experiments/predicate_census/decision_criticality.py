"""
Decision-criticality analysis.

For each gold30 scenario, asks the question precisely instead of counterfactually
flipping one fact at a time in isolation (the "synergy trap" -- a single-fact flip
can show "not critical" even when a fact matters only in combination with another
still-unknown fact, and single-fact testing can silently miss that).

Method: bracket the true extremes, not a sample point.
  1. Every predicate negated anywhere in the reachable graph (the full EDB
     negation list, data/negation_audit.json) is a "blocker" -- asserting it
     TRUE can only ever remove derivations (standard Datalog
     negation-as-failure monotonicity).
  2. Every other predicate in gold30's independent ground truth is an
     "enabler" -- since it is *never* negated anywhere in the reachable
     graph (the negation census is exhaustive, not sampled), asserting it
     TRUE can only ever add derivations, never remove any. This is a
     provable monotonicity guarantee, not a heuristic.
  3. For each scenario, build three fact sets from the same fixed base (given +
     the independently-confirmed-TRUE facts):
       V_closed:     nothing else asserted (closed-world default -- the
                      number reported as the baseline throughout this
                      project's evaluation).
       V_best_case:  + every UNKNOWN enabler asserted TRUE (blockers stay
                      absent) -- the true upper bound on how permission-friendly
                      this scenario could possibly be.
       V_worst_case: + every UNKNOWN blocker asserted TRUE (enablers stay
                      absent) -- the true lower bound.
     V_worst_case <= V_closed <= V_best_case is guaranteed by construction, not
     assumed.
  4. If V_worst_case == V_best_case, the scenario's verdict is provably invariant
     to *every* possible resolution of *every* UNKNOWN it contains -- none of them
     are decision-critical, and this conclusion required no per-fact testing at
     all, so there is no synergy trap to fall into.
  5. If V_worst_case != V_best_case, some UNKNOWN fact(s) genuinely matter. Only
     THEN drill down per-fact (now legitimate, since we already know from the
     bracket that this scenario is sensitive to something) to find which single
     facts are individually sufficient, and flag any case where no single fact
     is sufficient but the full set is (a genuine synergy).

Also reports V_closed vs. each scenario's expected_verdict -- a validity check on
the independent ground truth itself: how well does grounding on ONLY
independently-confirmed facts (no assumptions, no candidate-construction
convention) reproduce the corpus-author's intended answer.

Usage: python3 decision_criticality.py
Writes: ../../data/decision_criticality.json
"""
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOUFFLE_ROOT = ROOT / "souffle"


def parse_fact(line):
    line = line.split("//")[0].strip().rstrip(".")
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$', line)
    if not m:
        return None
    name, argstr = m.group(1), m.group(2)
    args = [a.strip().strip('"') for a in re.findall(r'"([^"]*)"', argstr)]
    return name, args


def fmt_fact(name, args):
    return f'{name}(' + ", ".join(f'"{a}"' for a in args) + ').'


def run_souffle(facts, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    dl = workdir / "scenario.dl"
    lines = ['#include "hipaa.dl"', ""] + [fmt_fact(n, a) for n, a in facts]
    lines += ["", ".output is_disclosure_allowed", ".output is_disclosure_denied"]
    dl.write_text("\n".join(lines))
    subprocess.run(["souffle", "-I", str(SOUFFLE_ROOT), "-D", str(workdir), str(dl)],
                    capture_output=True, text=True, timeout=60, cwd=SOUFFLE_ROOT)
    allowed = workdir / "is_disclosure_allowed.csv"
    denied = workdir / "is_disclosure_denied.csv"
    n_allowed = len(allowed.read_text().splitlines()) if allowed.exists() else 0
    n_denied = len(denied.read_text().splitlines()) if denied.exists() else 0
    return "PERMITTED" if n_allowed else "DENIED"


def main():
    bench = json.loads((ROOT / "data/athena_bench_gold30.json").read_text())
    gt = json.loads((ROOT / "data/gold30_independent_ground_truth.json").read_text())
    negation_audit = json.loads((ROOT / "data/negation_audit.json").read_text())
    blocker_names = {r["predicate"] for r in negation_audit if r["type"] == "EDB"}

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
            confirmed_true = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                               for r in rows if r["independent_value"] == "TRUE"]
            unknown_rows = [r for r in rows if r["independent_value"] == "UNKNOWN"]
            unknown_enablers = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                                 for r in unknown_rows if r["predicate"] not in blocker_names]
            unknown_blockers = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                                 for r in unknown_rows if r["predicate"] in blocker_names]

            base = given + confirmed_true
            v_closed = run_souffle(base, tmp / f"s{i}_closed")
            v_best = run_souffle(base + unknown_enablers, tmp / f"s{i}_best")
            v_worst = run_souffle(base + unknown_blockers, tmp / f"s{i}_worst")

            robust = (v_best == v_worst)
            critical_facts = []
            synergy = False

            if not robust:
                # drill down: which single enabler(s) alone flip v_closed -> PERMITTED
                for name, args in unknown_enablers:
                    v = run_souffle(base + [(name, args)], tmp / f"s{i}_e_{name}")
                    if v != v_closed:
                        critical_facts.append({"predicate": name, "args": args, "role": "enabler", "alone_sufficient": True})
                # which single blocker(s) alone flip v_closed -> DENIED
                for name, args in unknown_blockers:
                    v = run_souffle(base + [(name, args)], tmp / f"s{i}_b_{name}")
                    if v != v_closed:
                        critical_facts.append({"predicate": name, "args": args, "role": "blocker", "alone_sufficient": True})
                if not critical_facts:
                    # bracket showed sensitivity but no single fact alone reproduces it -- genuine synergy
                    synergy = True
                    critical_facts = [
                        {"predicate": n, "args": a, "role": "enabler" if n not in blocker_names else "blocker", "alone_sufficient": False}
                        for n, a in (unknown_enablers + unknown_blockers)
                    ]

            results.append({
                "scenario": sid,
                "expected_verdict": expected,
                "v_closed": v_closed,
                "v_best_case": v_best,
                "v_worst_case": v_worst,
                "closed_matches_expected": v_closed == expected,
                "robust_to_all_unknowns": robust,
                "n_unknown_total": len(unknown_rows),
                "n_unknown_enablers": len(unknown_enablers),
                "n_unknown_blockers": len(unknown_blockers),
                "synergy_detected": synergy,
                "critical_facts": critical_facts,
            })
            print(f"{sid}: expected={expected} closed={v_closed} best={v_best} worst={v_worst} "
                  f"robust={robust} n_unknown={len(unknown_rows)} critical={len(critical_facts)}"
                  f"{' [SYNERGY]' if synergy else ''}")

    (ROOT / "data/decision_criticality.json").write_text(json.dumps(results, indent=2))

    n = len(results)
    closed_match = sum(1 for r in results if r["closed_matches_expected"])
    robust_count = sum(1 for r in results if r["robust_to_all_unknowns"])
    total_unknown = sum(r["n_unknown_total"] for r in results)
    total_critical = sum(len(r["critical_facts"]) for r in results)
    synergy_count = sum(1 for r in results if r["synergy_detected"])

    print(f"\n=== SUMMARY (n={n}) ===")
    print(f"V_closed matches expected_verdict: {closed_match}/{n}")
    print(f"Scenarios robust to ALL their unknowns (verdict invariant): {robust_count}/{n}")
    print(f"Total UNKNOWN predicate-instances across all scenarios: {total_unknown}")
    print(f"Of those, individually/jointly decision-critical: {total_critical}")
    print(f"Scenarios with genuine synergy (no single fact sufficient, but the set is): {synergy_count}")


if __name__ == "__main__":
    main()
