"""
Atomic evidence acquisition for three predicates that are hard to answer
with a single direct TRUE/FALSE/UNKNOWN judgment: believes_minimum_necessary,
inconsistent_with_notice_of_privacy_practices, ce_wrongly_denied_access.

Earlier phrasings for these three still asked the model to make a
TRUE/FALSE/UNKNOWN judgment at each step. Here, for each predicate, ONE call
extracts every relevant evidence field as raw quoted text (or NONE_STATED),
with ZERO TRUE/FALSE/UNKNOWN judgments made by the model anywhere in the
call. All interpretation happens afterward, in one deterministic Python
function that looks at the whole evidence bundle at once rather than gating
sequentially. The oracle is asked, in one shot, "what does the narrative
explicitly say?", never "is this true?"

Validated against the same acceptance criteria used throughout this
project's oracle-phrasing development, per predicate, against the
production oracle's direct-question phrasing as the baseline:
  1. Unsupported-TRUE rate decreases.
  2. Correct TRUE/FALSE classification does not decrease materially.
  3. Retention accuracy does not decrease -- only claimed where a retention
     set exists at all (inconsistent_with_notice_of_privacy_practices has
     zero resolved instances anywhere in the gold30 benchmark, so no
     retention claim is possible or made for it).
  4/5. False-permit rate and search optimality are not measurable from this
       isolated harness and are not claimed here.

Usage: python3 atomic_acquisition.py --model gemma3:4b --provider ollama
Writes: ../../data/atomic_acquisition_e_<model>.json (or _f_ for --condition F)
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import run_baselines as base  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _atomic_extract(narrative, fields):
    """One call, all fields for one predicate, pure extraction -- no
    TRUE/FALSE/UNKNOWN from the model."""
    field_list = "\n".join(f"{name}: {desc}" for name, desc in fields)
    field_fmt = "\n".join(f"{name}: <span or NONE_STATED>" for name, _desc in fields)
    prompt = f"""You are extracting text from a narrative for a formal legal verifier. You are NOT making any judgment about what the text means -- only quoting text, or reporting that no such text exists.

NARRATIVE:
{narrative}

For each of the following, quote the exact SHORT span (a phrase, at most one sentence) of the narrative that matches, or respond exactly NONE_STATED if no such span exists:

{field_list}

Respond in exactly this format, one line per field, nothing else:
{field_fmt}
"""
    resp = base.ollama(prompt)
    evidence = {}
    for name, _desc in fields:
        m = re.search(rf'{name}:\s*(.*?)(?=\n[A-Z_]+:|\Z)', resp, re.S)
        val = m.group(1).strip() if m else ""
        cleaned = val.strip('"“”\' \n')
        if not cleaned or cleaned.upper().startswith("NONE_STATED") or len(cleaned) > 0.5 * len(narrative):
            evidence[name] = None
        else:
            evidence[name] = val
    return evidence


# ---------- believes_minimum_necessary ----------

BMN_FIELDS = [
    ("PURPOSE", "the exact span stating the purpose or reason for this disclosure/access"),
    ("NO_PURPOSE", "the exact span showing the discloser had NO legitimate purpose (e.g. curiosity, no assigned role, no treatment relationship) -- not a description of a purpose, but evidence of its absence"),
    ("DISCLOSED_ITEMS", "the exact span naming or describing the specific items of information actually disclosed or accessed"),
    ("EXCLUSION_LANGUAGE", "the exact span indicating specific items were NOT included, or that access/disclosure was RESTRICTED relative to a larger set -- e.g. 'only', 'limited to', 'excluding', 'basic', or a direct contrast with 'the complete/entire/full record'"),
    ("COMPLETE_RECORD_LANGUAGE", "the exact span stating the COMPLETE, ENTIRE, or FULL record/file was disclosed, with nothing withheld"),
]


def derive_believes_minimum_necessary(e):
    if e["NO_PURPOSE"] is not None:
        return "FALSE"
    if e["PURPOSE"] is None:
        return "UNKNOWN"
    if e["DISCLOSED_ITEMS"] is None:
        return "UNKNOWN"
    if e["EXCLUSION_LANGUAGE"] is not None:
        return "TRUE"
    if e["COMPLETE_RECORD_LANGUAGE"] is not None:
        return "FALSE"
    return "UNKNOWN"


# ---------- inconsistent_with_notice_of_privacy_practices ----------

NPP_FIELDS = [
    ("NOTICE_CONTENT", "the exact span stating what the covered entity's posted notice of privacy practices says, permits, or restricts"),
    ("CONFLICT_LANGUAGE", "the exact span stating or clearly implying this disclosure conflicts with, violates, or is inconsistent with the notice of privacy practices"),
    ("CONSISTENCY_LANGUAGE", "the exact span stating or clearly implying this disclosure is consistent with, permitted by, or matches the notice of privacy practices"),
]


def derive_inconsistent_with_notice(e):
    if e["NOTICE_CONTENT"] is None:
        return "UNKNOWN"
    if e["CONFLICT_LANGUAGE"] is not None:
        return "TRUE"
    if e["CONSISTENCY_LANGUAGE"] is not None:
        return "FALSE"
    return "UNKNOWN"


# ---------- ce_wrongly_denied_access ----------

CWD_FIELDS = [
    ("ACCESS_REQUEST", "the exact span showing the record-subject individual (or their personal representative) specifically requesting access to or a copy of THEIR OWN records"),
    ("NO_ACCESS_REQUEST", "the exact span indicating this narrative does NOT involve the record-subject individual (or their personal representative) requesting their own records at all -- e.g. it describes disclosure to or an inquiry by a different, unrelated third party, or a request for someone else's records rather than the requester's own"),
    ("DENIAL", "the exact span showing the covered entity denied that request"),
    ("VALID_GROUND", "the exact span stating a valid legal ground or basis for the denial"),
    ("NO_VALID_GROUND", "the exact span stating or clearly implying no valid legal ground existed for the denial"),
]


def derive_ce_wrongly_denied(e):
    """
    The original single ACCESS_REQUEST field defaulted to FALSE whenever
    absent, conflating "this narrative doesn't establish a request was made"
    with "this narrative establishes no request was made" -- the exact
    closed-world collapse this project exists to prevent. NO_ACCESS_REQUEST
    mirrors believes_minimum_necessary's NO_PURPOSE pattern: only derive
    FALSE from absence when the narrative affirmatively shows the
    predicate's precondition doesn't apply, not merely because nothing was
    extracted.
    """
    if e["NO_ACCESS_REQUEST"] is not None:
        return "FALSE"
    if e["ACCESS_REQUEST"] is None:
        return "UNKNOWN"
    if e["DENIAL"] is None:
        return "FALSE"
    if e["VALID_GROUND"] is not None:
        return "FALSE"
    if e["NO_VALID_GROUND"] is not None:
        return "TRUE"
    return "UNKNOWN"


# ---------- ce_wrongly_denied_access, condition F ----------
#
# The field-level trace shows NO_ACCESS_REQUEST -- not VALID_GROUND/
# NO_VALID_GROUND -- drives most primary-set wrong commits: the model
# quotes something as "no access request" on narratives that are simply
# off-topic for this predicate. The field is redundant with existing
# UNKNOWN handling: ACCESS_REQUEST absent already derives UNKNOWN.
# Condition F removes NO_ACCESS_REQUEST outright rather than reword it
# again -- this is a field-deletion, not another wording iteration.

CWD_FIELDS_F = [f for f in CWD_FIELDS if f[0] != "NO_ACCESS_REQUEST"]


def derive_ce_wrongly_denied_f(e):
    if e["ACCESS_REQUEST"] is None:
        return "UNKNOWN"
    if e["DENIAL"] is None:
        return "FALSE"
    if e["VALID_GROUND"] is not None:
        return "FALSE"
    if e["NO_VALID_GROUND"] is not None:
        return "TRUE"
    return "UNKNOWN"


PREDICATES = {
    "believes_minimum_necessary": (BMN_FIELDS, derive_believes_minimum_necessary),
    "inconsistent_with_notice_of_privacy_practices": (NPP_FIELDS, derive_inconsistent_with_notice),
    "ce_wrongly_denied_access": (CWD_FIELDS, derive_ce_wrongly_denied),
}

# predicates with an alternate (20B) condition, selected via --condition F
CONDITION_F = {
    "ce_wrongly_denied_access": (CWD_FIELDS_F, derive_ce_wrongly_denied_f),
}


def condition_e(name, narrative, fields, derive):
    evidence = _atomic_extract(narrative, fields)
    return derive(evidence), evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", required=True, choices=["ollama", "anthropic", "openai"])
    ap.add_argument("--predicate", choices=list(PREDICATES) + ["all"], default="all")
    ap.add_argument("--condition", choices=["E", "F"], default="E",
                     help="F selects the alternate field set where one exists")
    cli = ap.parse_args()
    base.MODEL, base.PROVIDER = cli.model, cli.provider
    model_slug = re.sub(r'[^a-z0-9]+', '_', cli.model.lower())
    cond_label = cli.condition

    bench = json.loads((ROOT / "data/athena_bench_gold30.json").read_text())
    narratives = {s["id"].replace("athena-", ""): s["narrative"]["situation"] + "\n\nQuestion: " + s["narrative"]["question"]
                  for s in bench["scenarios"]}
    gt = json.loads((ROOT / "data/gold30_independent_ground_truth.json").read_text())
    phase19 = json.loads((ROOT / f"data/gold30_oracle_interface_{model_slug}.json").read_text())

    target_predicates = list(PREDICATES) if cli.predicate == "all" else [cli.predicate]
    if cond_label == "F":
        target_predicates = [p for p in target_predicates if p in CONDITION_F]
    all_results = {}

    for pred_name in target_predicates:
        fields, derive = CONDITION_F[pred_name] if cond_label == "F" else PREDICATES[pred_name]
        primary = []
        for r in phase19:
            for c in r["oracle_calls"]:
                if c["predicate"] == pred_name:
                    primary.append({"scenario": r["scenario"], "args": c["args"],
                                     "ground_truth": c["ground_truth"], "A": c["llm_answer"]})

        retention = []
        for r in gt:
            if r["predicate"] == pred_name and r["independent_value"] in ("TRUE", "FALSE"):
                retention.append({"scenario": r["scenario"],
                                   "args": [a.strip() for a in r["args"].split(",")],
                                   "ground_truth": r["independent_value"]})

        print(f"\n### {pred_name} ###")
        print(f"Primary (critical) set: {len(primary)} queries")
        print(f"Retention (already-resolved) set: {len(retention)} queries"
              + ("  <-- none exist for this predicate in gold30; no retention claim possible" if not retention else ""))

        results = {"primary": [], "retention": []}
        for set_label, items in [("primary", primary), ("retention", retention)]:
            for item in items:
                narrative = narratives[item["scenario"]]
                value, evidence = condition_e(pred_name, narrative, fields, derive)
                row = {"scenario": item["scenario"], "args": item["args"],
                       "ground_truth": item["ground_truth"], cond_label: value,
                       f"{cond_label}_evidence": evidence}
                if "A" in item:
                    row["A"] = item["A"]
                results[set_label].append(row)
                a_str = f" A={item['A']}" if "A" in item else ""
                print(f"[{set_label}] {item['scenario']}: truth={item['ground_truth']}{a_str} {cond_label}={value}")

        all_results[pred_name] = results

        def metrics(items, key):
            n = len(items)
            if n == 0:
                return
            unk = sum(1 for r in items if r[key] == "UNKNOWN")
            t = sum(1 for r in items if r[key] == "TRUE")
            f = sum(1 for r in items if r[key] == "FALSE")
            correct = sum(1 for r in items if r[key] == r["ground_truth"])
            wrong = sum(1 for r in items if r[key] != r["ground_truth"] and r[key] != "UNKNOWN")
            print(f"  {key}: UNKNOWN={unk}/{n} TRUE={t}/{n} FALSE={f}/{n} correct={correct}/{n} ({correct/n:.1%}) wrong_commits={wrong}")

        print(f"--- SUMMARY ({pred_name}, {cli.model}) ---")
        print(f"Primary set (n={len(primary)}, ground truth always UNKNOWN):")
        metrics(results["primary"], "A")
        metrics(results["primary"], cond_label)
        if retention:
            print(f"Retention set (n={len(retention)}, ground truth TRUE/FALSE):")
            metrics(results["retention"], cond_label)

    suffix = "atomic_acquisition_e" if cond_label == "E" else "atomic_acquisition_f"
    out_path = ROOT / f"data/{suffix}_{model_slug}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
