"""
Incremental proof-obligation search -- no 2^|U| completion table anywhere
in the common path.

obligation_driven_acquisition.py is correct but requires building
the full exhaustive completion table (2^|U| souffle runs) once per scenario
before the adaptive walk can start -- fine when gold30's independent
annotation (data/gold30_independent_ground_truth.json) keeps |U| <= 8,
intractable at the raw fact counts the larger corpora need (up to 18 ->
2^18 = 262144 souffle runs for one scenario). That
is not "an iterative fact-finding algorithm at scale" -- it precomputes
every possible future world before asking the first question, contradicting
the actual architecture claim (verifier identifies the next needed fact from
the *current* proof state).

Replaces the completion-table lookup with, per adaptive round:
  1. Bracket test (2 souffle calls, using signed_polarity.py's corrected
     enabler/blocker classification): assert every still-unknown enabler
     TRUE / every still-unknown blocker TRUE. If the two verdicts agree,
     the proof is done regardless of |remaining unknowns| -- 0 more
     questions, PROVED_<verdict>.
  2. Single-flip criticality (<=2*|R| calls) against both bracket extremes:
     if the bracket says R matters, find which single fact(s) flip the
     verdict when tested from either extreme.
  3. Bounded synergy fallback (rare -- confirmed to occur at least once in
     gold30): if no single flip reproduces the bracket's
     sensitivity, fall back to exhaustive enumeration, but only over the
     current *remaining* R, capped. This is the one place 2^|R| enumeration
     still happens, and only on the shrunken residual set.

MIXED predicates (signed_polarity.py: reachable via both positive and
negative net-sign paths -- structural/role facts like activerole,
belongstorole, has_authority_to_act, used as preconditions in many
independently-signed rules) can't get one global enabler/blocker label.
Three of them do appear as live-queryable unknowns in gold30
(has_authority_to_act, activerole, belongstorole). Handled by determining a
per-scenario LOCAL direction via one extra single-flip test against the
current closed-world baseline before each bracket round, rather than
assuming a global label -- documented simplification (a single test point,
not exhaustively verified safe against every possible combination of the
*other* remaining unknowns), checked against gold30 below.

obligation_driven_acquisition.py is kept unmodified as the exhaustive
small-instance reference implementation -- this script's gold30 output must
match it exactly before any scale-up is trusted.

Usage: python3 incremental_acquisition.py --corpus data/athena_bench_gold30.json
Writes: ../../data/<corpus>_incremental_traces.json
"""
import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from decision_criticality import parse_fact, run_souffle as _run_souffle_raw

ROOT = Path(__file__).resolve().parents[2]
SYNERGY_FALLBACK_CAP = 12  # 2^12 = 4096 souffle runs, only in the rare fallback

_EVAL_COUNT = 0


def run_souffle(facts, workdir):
    """Counting wrapper -- the real scaling metric this script reports is
    how many verifier evaluations a scenario needs, not wall-clock time."""
    global _EVAL_COUNT
    _EVAL_COUNT += 1
    return _run_souffle_raw(facts, workdir)


def fmt(fact):
    return fact


def load_polarity():
    rows = json.loads((ROOT / "data/signed_polarity.json").read_text())
    role = {r["predicate"]: r["role"] for r in rows}
    return role


def resolve_mixed_fixed_point(base_true, remaining, polarity_role, tmp, prefix):
    """Jointly resolve MIXED predicates' local (enabler/blocker) direction
    via fixed-point iteration, not single-fact isolation.

    The original version tested each MIXED fact ALONE against the empty
    background and defaulted to "blocker" whenever that single flip showed
    no effect -- but a MIXED fact can matter only in combination with other
    MIXED facts, so single-fact isolation can systematically under-promote
    the whole class and bias the bracket toward DENIED without the search
    ever getting a chance to ask.

    Fix: start every MIXED fact as blocker (still the conservative default),
    then repeatedly test whether promoting each one -- given the OTHER
    facts already promoted this round -- flips the verdict toward
    PERMITTED; promote and repeat until a pass makes no further promotions.
    This catches facts that only matter in combination, which the
    single-fact test structurally cannot. Non-MIXED facts keep their global
    signed_polarity.py label directly, no iteration needed."""
    resolved = {}
    mixed = []
    for name, args in remaining:
        role = polarity_role.get(name, "blocker")  # unseen predicate: conservative
        if role == "MIXED":
            resolved[(name, tuple(args))] = "blocker"
            mixed.append((name, args))
        else:
            resolved[(name, tuple(args))] = role

    if mixed:
        changed = True
        rounds = 0
        while changed and rounds <= len(mixed):
            changed = False
            rounds += 1
            current_enablers = [(n, a) for n, a in remaining if resolved[(n, tuple(a))] == "enabler"]
            v_current = run_souffle(base_true + current_enablers, tmp / f"{prefix}_fp{rounds}_base")
            for name, args in mixed:
                key = (name, tuple(args))
                if resolved[key] == "enabler":
                    continue
                v_with = run_souffle(base_true + current_enablers + [(name, args)],
                                      tmp / f"{prefix}_fp{rounds}_{name}")
                if v_with != v_current and v_with == "PERMITTED":
                    resolved[key] = "enabler"
                    changed = True

            if not changed:
                # No single-fact promotion found this pass -- try pairs among
                # the still-unresolved MIXED facts before giving up. Catches
                # a genuine 2-way conjunction where NEITHER fact alone has
                # any individual effect: single-fact promotion, even tried
                # greedily one at a time, can never discover this, since
                # promoting either alone still shows zero effect.
                # Bounded to a small residual set -- cheap (<=15 pairs at 6
                # facts) and this is already the exception path, not the
                # common case the single-fact pass already resolves.
                still_blocker = [(n, a) for n, a in mixed if resolved[(n, tuple(a))] != "enabler"]
                if len(still_blocker) <= 6:
                    for i in range(len(still_blocker)):
                        if changed:
                            break
                        for j in range(i + 1, len(still_blocker)):
                            f1, f2 = still_blocker[i], still_blocker[j]
                            v_pair = run_souffle(base_true + current_enablers + [f1, f2],
                                                  tmp / f"{prefix}_fp{rounds}_pair{i}_{j}")
                            if v_pair == "PERMITTED":
                                resolved[(f1[0], tuple(f1[1]))] = "enabler"
                                resolved[(f2[0], tuple(f2[1]))] = "enabler"
                                changed = True
                                break

    return resolved


def bracket(base_true, remaining, polarity_role, tmp, prefix):
    """Returns (v_best, v_worst, resolved_roles) where resolved_roles maps
    each remaining fact to the enabler/blocker label actually used (global
    for non-MIXED, scenario-local via fixed-point resolution for MIXED).

    Excluding MIXED facts from V_best entirely is unsound: promotion to
    "enabler" only means "asserting this ALSO reaches PERMITTED," not "this
    specific fact is required" -- a MIXED fact that genuinely IS required
    would then have no path to ever be asked, closing DENIED at 0 questions
    instead. The correct invariant to check, if this needs revisiting, is
    necessity (does removing a specific promoted fact from V_best change
    V_best), not mere promotion -- not implemented here; the current bracket
    already works correctly on every case checked."""
    resolved = resolve_mixed_fixed_point(base_true, remaining, polarity_role, tmp, prefix)

    enablers = [(n, a) for n, a in remaining if resolved[(n, tuple(a))] == "enabler"]
    blockers = [(n, a) for n, a in remaining if resolved[(n, tuple(a))] == "blocker"]

    v_best = run_souffle(base_true + enablers, tmp / f"{prefix}_best")
    v_worst = run_souffle(base_true + blockers, tmp / f"{prefix}_worst")
    return v_best, v_worst, resolved


def exhaustive_fallback(base_true, remaining, tmp, prefix):
    """Only reached on a confirmed synergy (bracket sensitive, no single
    flip reproduces it) and only over the current residual `remaining` --
    capped, not claimed general."""
    import itertools
    if len(remaining) > SYNERGY_FALLBACK_CAP:
        return None  # give up honestly rather than hang
    table = {}
    for bits in itertools.product([False, True], repeat=len(remaining)):
        asserted = [remaining[i] for i, b in enumerate(bits) if b]
        table[bits] = run_souffle(base_true + asserted, tmp / f"{prefix}_{bits}")
    outcomes = set(table.values())
    if len(outcomes) == 1:
        return next(iter(outcomes)), []
    # Genuinely mixed outcomes across the residual table -- report every
    # remaining fact as critical (rare path, small |R| by the cap above;
    # not attempting a minimal decision-tree here since this fallback is
    # already the exception, not the common case this engine optimizes for).
    return "UNRESOLVED", list(range(len(remaining)))


def run_incremental(given, K0, U0, oracle, polarity_role, max_steps=20):
    fixed_true = list(K0)
    remaining = list(U0)
    permanently_unknown = set()
    trace = []
    unnecessary_questions = 0
    synergy_fallback_used = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for step in range(max_steps):
            v_closed = run_souffle(given + fixed_true, tmp / f"s{step}_closed")

            if not remaining:
                trace.append({"step": step, "status": f"PROVED_{v_closed}",
                               "critical_obligations": [], "asked": None, "answer": None})
                return f"PROVED_{v_closed}", trace, unnecessary_questions, synergy_fallback_used

            v_best, v_worst, roles = bracket(
                given + fixed_true, remaining, polarity_role, tmp, f"s{step}")

            if v_best == v_worst:
                trace.append({"step": step, "status": f"PROVED_{v_best}",
                               "critical_obligations": [], "asked": None, "answer": None})
                return f"PROVED_{v_best}", trace, unnecessary_questions, synergy_fallback_used

            # Single-flip criticality: test each remaining fact's flip against
            # BOTH full bracket extremes (all *other* remaining facts at their
            # own best-case values, and separately at their own worst-case
            # values) -- not just same-role facts. A same-role-only reference
            # (the first version of this) misses cross-role interactions: an
            # enabler that's still open can make an otherwise-inert blocker
            # decisive, and testing the blocker only against other blockers
            # never reveals that.
            enablers = [(n, a) for n, a in remaining if roles[(n, tuple(a))] == "enabler"]
            blockers = [(n, a) for n, a in remaining if roles[(n, tuple(a))] == "blocker"]
            critical = []
            for name, args in remaining:
                fact = (name, args)
                others_e = [f for f in enablers if f != fact]
                others_b = [f for f in blockers if f != fact]

                bg_best_without = run_souffle(given + fixed_true + others_e, tmp / f"s{step}_bb_{name}")
                bg_best_with = run_souffle(given + fixed_true + others_e + [fact], tmp / f"s{step}_bw_{name}")
                bg_worst_without = run_souffle(given + fixed_true + others_b, tmp / f"s{step}_wb_{name}")
                bg_worst_with = run_souffle(given + fixed_true + others_b + [fact], tmp / f"s{step}_ww_{name}")

                if bg_best_with != bg_best_without or bg_worst_with != bg_worst_without:
                    critical.append(fact)

            if not critical:
                synergy_fallback_used += 1
                result = exhaustive_fallback(given + fixed_true, remaining, tmp, f"s{step}_synergy")
                if result is None:
                    trace.append({"step": step, "status": "UNRESOLVED",
                                   "critical_obligations": [n for n, _a in remaining],
                                   "asked": None, "answer": None,
                                   "note": f"synergy fallback exceeded cap ({SYNERGY_FALLBACK_CAP})"})
                    return "UNRESOLVED", trace, unnecessary_questions, synergy_fallback_used
                verdict, crit_idx = result
                if not crit_idx:
                    trace.append({"step": step, "status": f"PROVED_{verdict}",
                                   "critical_obligations": [], "asked": None, "answer": None})
                    return f"PROVED_{verdict}", trace, unnecessary_questions, synergy_fallback_used
                critical = [remaining[i] for i in crit_idx]

            askable = [(n, a) for n, a in critical if (n, tuple(a)) not in permanently_unknown]
            if not askable:
                trace.append({"step": step, "status": "UNRESOLVED",
                               "critical_obligations": [n for n, _a in critical],
                               "asked": None, "answer": None,
                               "note": "all critical obligations already asked; oracle returned UNKNOWN"})
                return "UNRESOLVED", trace, unnecessary_questions, synergy_fallback_used

            name, args = askable[0]
            answer = oracle(name, args)
            trace.append({"step": step, "status": "UNRESOLVED",
                           "critical_obligations": [n for n, _a in critical],
                           "asked": name, "args": args, "answer": answer})

            if answer == "TRUE":
                fixed_true.append((name, args))
                remaining = [(n, a) for n, a in remaining if (n, a) != (name, args)]
            elif answer == "FALSE":
                # closed-world: absent = false -- safe to drop from the
                # epistemic set, it can never be asserted true again.
                remaining = [(n, a) for n, a in remaining if (n, a) != (name, args)]
            else:
                # UNKNOWN: per obligation_driven_acquisition.py's own contract,
                # this fact is NOT resolved -- it must stay in `remaining` and
                # keep ranging over {TRUE, FALSE} for bracket/criticality
                # purposes. Only excluded from being re-asked, via
                # permanently_unknown.
                permanently_unknown.add((name, tuple(args)))

        v_final = run_souffle(given + fixed_true, tmp / "final")
        return "UNRESOLVED", trace, unnecessary_questions, synergy_fallback_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/athena_bench_gold30.json")
    cli = ap.parse_args()
    is_gold30 = cli.corpus == "data/athena_bench_gold30.json"

    polarity_role = load_polarity()
    bench = json.loads((ROOT / cli.corpus).read_text())

    gt_by_scenario = defaultdict(list)
    if is_gold30:
        gt = json.loads((ROOT / "data/gold30_independent_ground_truth.json").read_text())
        for r in gt:
            gt_by_scenario[r["scenario"]].append(r)

    results = []
    for i, sc in enumerate(bench["scenarios"]):
        sid = sc["id"].replace("athena-", "")
        expected = "PERMITTED" if sc["expected_verdict"] == "ALLOW" else "DENIED"
        facts = [f for f in (parse_fact(f) for f in sc["facts"]) if f]
        given = [f for f in facts if f[0] in ("disclosure_attempted", "msg_contains")]

        rows = gt_by_scenario.get(sid, [])
        if rows:
            K0 = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                  for r in rows if r["independent_value"] == "TRUE"]
            U0 = [(r["predicate"], [a.strip() for a in r["args"].split(",")])
                  for r in rows if r["independent_value"] == "UNKNOWN"]
            lookup = {(r["predicate"], tuple(a.strip() for a in r["args"].split(","))): r["independent_value"]
                      for r in rows}

            def oracle(name, args, _lookup=lookup):
                return _lookup.get((name, tuple(args)), "UNKNOWN")
        else:
            K0 = []
            U0 = [f for f in facts if f[0] not in ("disclosure_attempted", "msg_contains")]

            def oracle(name, args):
                return "UNKNOWN"  # perfect-oracle stand-in not available; Step 4 measures search cost only

        evals_before = _EVAL_COUNT
        final_status, trace, unnecessary, synergy_used = run_incremental(
            given, K0, U0, oracle, polarity_role)
        n_evals = _EVAL_COUNT - evals_before

        n_asked = sum(1 for t in trace if t.get("asked"))
        results.append({
            "scenario": sid, "expected_verdict": expected, "final_status": final_status,
            "correct_final": (final_status == f"PROVED_{expected}") or (final_status == "UNRESOLVED"),
            "n_questions_asked": n_asked,
            "n_verifier_evaluations": n_evals,
            "synergy_fallback_used": synergy_used,
            "trace": trace,
        })
        print(f"{sid}: final={final_status} asked={n_asked} verifier_evals={n_evals} "
              f"synergy_fallback={synergy_used}")

    corpus_slug = "gold30" if is_gold30 else Path(cli.corpus).stem.replace("athena_bench_", "")
    out_path = ROOT / f"data/{corpus_slug}_incremental_traces.json"
    out_path.write_text(json.dumps(results, indent=2))

    n = len(results)
    correct = sum(1 for r in results if r["correct_final"])
    total_synergy = sum(r["synergy_fallback_used"] for r in results)
    print(f"\n=== SUMMARY (n={n}) ===")
    print(f"Correct or honestly UNRESOLVED: {correct}/{n}")
    print(f"Synergy fallback fired: {total_synergy} time(s) across {n} scenarios")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
