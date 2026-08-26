"""
Baseline A (direct LLM verdict) + Baseline B (batched extraction,
closed-world and open-world verdict read), all 30 gold30 scenarios.

Usage:
    python3 run_baselines.py --model gemma3:4b --provider ollama
    python3 run_baselines.py --model llama3.1:8b --provider ollama
    python3 run_baselines.py --model claude-sonnet-5 --provider anthropic
"""
import argparse
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOUFFLE_ROOT = REPO / "souffle"
GOLD_PATH = REPO / "data/athena_bench_gold30.json"

# Only 4 of the 9 top-level guards in hipaa_top.dl are genuinely EDB (atomic
# oracle facts, safe to query as a single yes/no claim) -- the other 5 are
# IDB-derived compound relations (1-3 rules each, up to a 3-way disjunction of
# 5-condition conjunctions for require_authorization_by_164_508). Asking an LLM to
# resolve a compound derived relation in one shot is a different, harder task than
# resolving an atomic fact, and naming all 9 as "guards" without this distinction
# would conflate that difficulty gap with a genuine polarity/phrasing effect.
#
# Fix: replaced each IDB guard with its real EDB sub-facts (verified against each
# .decl's actual argument order and arity, not assumed). hipaa.dl's own rules
# derive the compound guard automatically from these once asserted -- the verifier
# resolves IDB, the query layer only ever resolves EDB, per 02 SS2/03 SS1's binding rule.
GUARDS = [
    # -- Genuinely EDB, safe to query directly --
    ("lacks_authorized_relationship", ["p1", "p2", "q", "t"]),
    ("inconsistent_with_notice_of_privacy_practices", ["p1", "p2", "q", "m", "t", "u"]),
    ("ce_wrongly_denied_access", ["p1", "p2", "q", "t"]),
    ("reidentification_code_violates_514c", ["m", "t"]),
    # -- Decomposed from sent_via_non_requested_channel (was IDB) --
    ("requested_confidential_communication", ["p1", "q"]),
    ("accommodated_via_requested_channel", ["m", "p1", "q"]),
    # -- Decomposed from violates_restriction_agreement (was IDB, two levels deep) --
    ("requested_restriction", ["p1", "q", "t", "u"]),
    ("agreed_to_restriction", ["p1", "q", "t", "u"]),
    ("restriction_terminated", ["p1", "q", "t", "u"]),
    ("paid_out_of_pocket_in_full", ["q", "t"]),
    # -- Decomposed from violates_underwriting_restriction (was IDB) --
    ("received_phi_for_underwriting", ["p1", "q", "t"]),
    ("health_insurance_placed_with", ["q", "p1"]),
    ("is_required_by_law", ["p1", "p2", "q", "t", "u"]),
    # -- Decomposed from uses_genetic_info_for_underwriting (was IDB) --
    ("is_medical_appropriateness_determination", ["u"]),
    # -- Decomposed from require_authorization_by_164_508 (was IDB, 3-way disjunction) --
    ("obtained_authorization_164_508", ["p1", "p2", "q", "t", "u"]),
    ("exception_508a2", ["p1", "p2", "q", "m", "t", "u"]),
    ("exception_508a3", ["p1", "p2", "q", "m", "t", "u"]),
    ("receives_remuneration_for_phi", ["p1", "p2", "q", "t", "u"]),
]

# --- Model dispatch: one call() function, provider selected at startup ---
MODEL = None
PROVIDER = None
_ANTHROPIC_KEY = None
_OPENAI_KEY = None


def _load_env_key(var_name, cache_attr):
    cached = globals()[cache_attr]
    if cached is not None:
        return cached
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{var_name}="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            globals()[cache_attr] = key
            return key
    raise RuntimeError(f"{var_name} not found in .env")


def _call_ollama(prompt: str) -> str:
    payload = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        d = json.loads(resp.read())
    return d.get("response", "").strip()


def _call_anthropic(prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL, "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=payload,
        headers={"x-api-key": _load_env_key("ANTHROPIC_API_KEY", "_ANTHROPIC_KEY"),
                 "anthropic-version": "2023-06-01", "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read())
    return "".join(b.get("text", "") for b in d.get("content", [])).strip()


def _call_openai(prompt: str) -> str:
    # Newer reasoning-family models (gpt-5.x) reject temperature != 1 outright,
    # unlike every other provider used in this project -- omit it and accept
    # the model's default rather than force an unsupported value.
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {_load_env_key('OPENAI_API_KEY', '_OPENAI_KEY')}",
                 "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read())
    return d["choices"][0]["message"]["content"].strip()


def _dispatch(prompt: str) -> str:
    if PROVIDER == "anthropic":
        return _call_anthropic(prompt)
    if PROVIDER == "openai":
        return _call_openai(prompt)
    return _call_ollama(prompt)


def ollama(prompt: str, _max_retries: int = 8) -> str:
    """Dispatches to the configured backend (Ollama or OpenAI) by PROVIDER --
    the single call-site every oracle script in this project imports.

    Retries transient network failures (read timeouts, connection resets, rate
    limits) with exponential backoff -- found the hard way (2026-08-21): a
    single 60s read timeout on one call killed an entire unattended
    50-scenario background run with no way to resume mid-corpus. A second,
    sustained round of HTTP 429s later killed a 132-scenario run at 82/132
    even with retries, because the original 4-retry/max-4s-wait budget wasn't
    long enough for a real rate-limit window to clear -- HTTP 429 now gets a
    longer, separate backoff ceiling (up to 60s) and respects a Retry-After
    header when the API provides one, rather than sharing the same short
    budget as a one-off connection blip. Does not retry non-network errors
    (bad API key, malformed request) -- those fail the same way every time."""
    import socket
    import time
    import urllib.error

    last_exc = None
    for attempt in range(_max_retries):
        try:
            return _dispatch(prompt)
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code != 429 or attempt >= _max_retries - 1:
                raise
            retry_after = e.headers.get("Retry-After") if e.headers else None
            wait = float(retry_after) if retry_after else min(60, 5 * (2 ** attempt))
            print(f"  [rate limited (429) -- retry {attempt + 1}/{_max_retries} in {wait}s]", flush=True)
            time.sleep(wait)
        except (TimeoutError, socket.timeout, urllib.error.URLError, ConnectionError) as e:
            last_exc = e
            if attempt < _max_retries - 1:
                wait = min(30, 2 ** attempt)
                print(f"  [transient network error: {e!r} -- retry {attempt + 1}/{_max_retries} in {wait}s]",
                      flush=True)
                time.sleep(wait)
    raise last_exc


def parse_fact(line: str):
    line = line.split("//")[0].strip().rstrip(".")
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$', line)
    if not m:
        return None
    name, argstr = m.group(1), m.group(2)
    args = [a.strip().strip('"') for a in re.findall(r'"([^"]*)"', argstr)]
    return name, args


def build_scenario(sc):
    facts = [parse_fact(f) for f in sc["facts"]]
    facts = [f for f in facts if f]
    given = [f for f in facts if f[0] in ("disclosure_attempted", "msg_contains")]
    candidates = [f for f in facts if f[0] not in ("disclosure_attempted", "msg_contains")]

    da = next(f for f in facts if f[0] == "disclosure_attempted")
    p1, p2, q, m, t, u = da[1]
    binding = {"p1": p1, "p2": p2, "q": q, "m": m, "t": t, "u": u}

    seen = {(name, tuple(args)) for name, args in candidates}
    for gname, gargs in GUARDS:
        instantiated = tuple(binding[a] for a in gargs)
        key = (gname, instantiated)
        if key not in seen:
            candidates.append((gname, list(instantiated)))
            seen.add(key)

    return given, candidates, binding


def fmt_fact(name, args):
    return f'{name}(' + ", ".join(f'"{a}"' for a in args) + ').'


def extraction_prompt(narrative, given, candidates):
    given_txt = "\n".join(fmt_fact(n, a) for n, a in given)
    q_txt = "\n".join(f"{i+1}. {fmt_fact(n, a)}" for i, (n, a) in enumerate(candidates))
    return f"""You are extracting candidate facts from a narrative to feed a formal legal verifier. You are NOT making a legal judgment -- only reporting, for each numbered claim below, whether the narrative establishes it.

NARRATIVE:
{narrative}

CONTEXT (already established, given to you, do not re-answer):
{given_txt}

For EACH of the following numbered claims, decide TRUE, FALSE, or UNKNOWN. TRUE = narrative directly states or clearly implies it. FALSE = narrative directly states or clearly implies the opposite. UNKNOWN = narrative does not address it, or is too ambiguous. Do not guess -- if unsure, say UNKNOWN.

{q_txt}

Respond with exactly one line per claim, in this format, nothing else:
<number>: <TRUE|FALSE|UNKNOWN>
"""


def parse_extraction(text, n):
    out = {}
    for line in text.splitlines():
        m = re.match(r'^\s*(\d+)\s*[:.\)]\s*(TRUE|FALSE|UNKNOWN)', line.strip(), re.I)
        if m:
            out[int(m.group(1))] = m.group(2).upper()
    return out


def run_souffle(facts, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    dl = workdir / "scenario.dl"
    lines = ['#include "hipaa.dl"', ""]
    lines += [fmt_fact(n, a) for n, a in facts]
    lines += ["", ".output is_disclosure_allowed", ".output is_disclosure_denied"]
    dl.write_text("\n".join(lines))
    subprocess.run(["souffle", "-I", str(SOUFFLE_ROOT), "-D", str(workdir), str(dl)],
                    capture_output=True, text=True, timeout=60, cwd=SOUFFLE_ROOT)
    allowed = (workdir / "is_disclosure_allowed.csv")
    denied = (workdir / "is_disclosure_denied.csv")
    n_allowed = len(allowed.read_text().splitlines()) if allowed.exists() else 0
    n_denied = len(denied.read_text().splitlines()) if denied.exists() else 0
    return n_allowed, n_denied


def baseline_a(narrative_text):
    prompt = f"""You are answering a HIPAA disclosure-permission question based only on the narrative below and your own general knowledge of HIPAA. No other tools or documents are available to you.

NARRATIVE:
{narrative_text}

Answer with exactly one of: PERMITTED, DENIED, UNRESOLVED (use UNRESOLVED if you are not confident either way), followed by one sentence of reasoning.

Respond in exactly this format:
VERDICT: <PERMITTED|DENIED|UNRESOLVED>
REASON: <one sentence>
"""
    resp = ollama(prompt)
    m = re.search(r'VERDICT:\s*(PERMITTED|DENIED|UNRESOLVED)', resp, re.I)
    verdict = m.group(1).upper() if m else "UNRESOLVED"
    return verdict, resp


def main():
    global MODEL, PROVIDER
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g. gemma3:4b, llama3.1:8b, claude-sonnet-5")
    ap.add_argument("--provider", required=True, choices=["ollama", "anthropic"])
    args = ap.parse_args()
    MODEL, PROVIDER = args.model, args.provider
    results_path = Path(f"results_{re.sub(r'[^a-z0-9]+', '_', MODEL.lower())}.json")

    d = json.loads(GOLD_PATH.read_text())
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, sc in enumerate(d["scenarios"]):
            sid = sc["id"]
            gold = "PERMITTED" if sc["expected_verdict"] == "ALLOW" else "DENIED"
            narrative_text = sc["narrative"]["situation"] + "\n\nQuestion: " + sc["narrative"]["question"]

            a_verdict, a_raw = baseline_a(narrative_text)

            given, candidates, binding = build_scenario(sc)
            ext_prompt = extraction_prompt(narrative_text, given, candidates)
            ext_resp = ollama(ext_prompt)
            parsed = parse_extraction(ext_resp, len(candidates))

            true_facts = list(given)
            unknown_count = 0
            unknown_guards = 0
            guard_names = {g[0] for g in GUARDS}
            for idx, (name, args_) in enumerate(candidates, start=1):
                val = parsed.get(idx, "UNKNOWN")
                if val == "TRUE":
                    true_facts.append((name, args_))
                elif val == "UNKNOWN":
                    unknown_count += 1
                    if name in guard_names:
                        unknown_guards += 1

            n_allowed, n_denied = run_souffle(true_facts, tmp / f"s{i}")
            closed_verdict = "PERMITTED" if n_allowed else "DENIED"  # closed-world: never UNRESOLVED

            if n_allowed:
                open_verdict = "UNRESOLVED" if unknown_guards > 0 else "PERMITTED"
            else:
                open_verdict = "UNRESOLVED" if unknown_count > 0 else "DENIED"

            results.append({
                "id": sid, "gold": gold,
                "baseline_a": a_verdict,
                "baseline_b_closed": closed_verdict,
                "baseline_b_open": open_verdict,
                "n_candidates": len(candidates), "n_unknown": unknown_count, "n_unknown_guards": unknown_guards,
            })
            print(f"{sid}: gold={gold} A={a_verdict} B-closed={closed_verdict} B-open={open_verdict} "
                  f"(candidates={len(candidates)}, unknown={unknown_count}, unknown_guards={unknown_guards})")

    results_path.write_text(json.dumps(results, indent=2))

    def score(key):
        tp = sum(1 for r in results if r[key] == r["gold"])
        false_permit = sum(1 for r in results if r["gold"] == "DENIED" and r[key] == "PERMITTED")
        unresolved = sum(1 for r in results if r[key] == "UNRESOLVED")
        return tp, false_permit, unresolved

    print(f"\n=== SUMMARY ({MODEL}, n=30) ===")
    for key in ["baseline_a", "baseline_b_closed", "baseline_b_open"]:
        tp, fp, unr = score(key)
        print(f"{key}: correct={tp}/30, false_PERMIT={fp}, UNRESOLVED={unr}")


if __name__ == "__main__":
    main()
