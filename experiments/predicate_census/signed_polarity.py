"""
Signed reachability -- correct enabler/blocker classification.

decision_criticality.py's V_best/V_worst bracket (and incremental_acquisition.py's
incremental engine, which reuses that bracket as its core per-round check)
needs to know, for each EDB predicate, whether asserting it TRUE can only
ever ADD derivations of is_disclosure_allowed (an "enabler," safe to assert
in the best-case bracket) or only ever REMOVE them (a "blocker," safe to
assert in the worst-case bracket).

decision_criticality.py's existing test -- "negated anywhere in the
reachable graph" (data/negation_audit.json's 28-row EDB list) -- gets this
wrong whenever a predicate is negated on the path to an IDB predicate that
is *itself* negated again further up: two negations cancel.
obtained_authorization_164_508 is negated inside
require_authorization_by_164_508's definition (making it look like a
"blocker" to the naive test), but require_authorization_by_164_508 is
itself negated in is_disclosure_allowed's top-level clause -- net effect,
asserting obtained_authorization_164_508 TRUE only ever helps
is_disclosure_allowed derive. It's a net enabler.

Fix: walk the *signed* dependency graph from is_disclosure_allowed down to
every EDB predicate, multiplying edge signs (+1 for a positive body atom,
-1 for a negated one) along every path, and classify by the *net* sign
reaching is_disclosure_allowed -- not by whether a negation occurs anywhere
in isolation. A predicate reached via paths of both signs is flagged MIXED,
not silently assigned one label.

Reuses build_census.py's statement parser (paren-depth/string-aware, since
a naive line parser was found to silently drop multi-line clauses) rather
than re-parsing the .dl files with anything simpler.

Usage: python3 signed_polarity.py
Writes: ../../data/signed_polarity.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from build_census import (
    ROOT, FILES, extract_decls, strip_comments, remove_directives, split_clauses,
)

ROOT_PREDICATE = "is_disclosure_allowed"


def extract_signed_rules(files, decls):
    """rules[head] = [(filename, [(body_pred, sign), ...]), ...] -- one entry
    per clause (OR-branch), sign = +1 for a positive atom, -1 for `!atom`."""
    rules = defaultdict(list)
    call_re = re.compile(r'(!?)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')
    for fn in files:
        text = remove_directives(strip_comments(fn.read_text()))
        for clause in split_clauses(text):
            hm = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', clause)
            if not hm or hm.group(1) not in decls or ":-" not in clause:
                continue
            head = hm.group(1)
            body = clause.split(":-", 1)[1]
            occurrences = []
            for neg, name in call_re.findall(body):
                if name == head or name not in decls:
                    continue
                occurrences.append((name, -1 if neg else 1))
            rules[head].append((str(fn.relative_to(ROOT)), occurrences))
    return rules


def signed_reachability(rules, root):
    """Returns {predicate: set(net_signs reaching root)}, via DFS from root
    down into body predicates, multiplying signs along each path. Guards
    against cycles (none expected in this stratified formalization, but not
    assumed) by capping recursion depth rather than looping forever."""
    signs_reaching = defaultdict(set)
    signs_reaching[root].add(1)

    def visit(pred, sign_from_root, path):
        if pred in path or len(path) > 200:
            return  # cycle guard -- report, don't hang
        for _fn, occurrences in rules.get(pred, []):
            for body_pred, local_sign in occurrences:
                net = sign_from_root * local_sign
                if net not in signs_reaching[body_pred]:
                    signs_reaching[body_pred].add(net)
                    visit(body_pred, net, path | {pred})

    visit(root, 1, frozenset())
    return signs_reaching


def main():
    decls, _arity = extract_decls(FILES)
    rules = extract_signed_rules(FILES, decls)
    idb_names = set(rules.keys())
    edb_names = set(decls.keys()) - idb_names

    signs_reaching = signed_reachability(rules, ROOT_PREDICATE)

    rows = []
    for name in sorted(edb_names & set(signs_reaching.keys())):
        signs = signs_reaching[name]
        if signs == {1}:
            role = "enabler"
        elif signs == {-1}:
            role = "blocker"
        else:
            role = "MIXED"
        rows.append({"predicate": name, "net_signs": sorted(signs), "role": role})

    out = {r["predicate"]: r for r in rows}
    (ROOT / "data/signed_polarity.json").write_text(json.dumps(rows, indent=2))

    mixed = [r for r in rows if r["role"] == "MIXED"]
    print(f"Reachable EDB predicates classified: {len(rows)}")
    print(f"  enablers: {sum(1 for r in rows if r['role'] == 'enabler')}")
    print(f"  blockers: {sum(1 for r in rows if r['role'] == 'blocker')}")
    print(f"  MIXED (context-dependent, needs per-path handling): {len(mixed)}")
    if mixed:
        print(f"    -> {[r['predicate'] for r in mixed]}")

    # Validation against the confirmed cases before trusting this output.
    checks = {
        "obtained_authorization_164_508": "enabler",
        "lacks_authorized_relationship": "blocker",
        "inconsistent_with_notice_of_privacy_practices": "blocker",
        "ce_wrongly_denied_access": "blocker",
    }
    print("\n=== Validation against known cases ===")
    all_ok = True
    for pred, expected in checks.items():
        got = out.get(pred, {}).get("role", "NOT FOUND")
        ok = got == expected
        all_ok &= ok
        print(f"  {pred}: expected={expected} got={got} {'OK' if ok else '*** MISMATCH ***'}")
    if not all_ok:
        print("\n*** at least one validation check failed -- do not trust this "
              "classification for the bracket engine until resolved ***")


if __name__ == "__main__":
    main()
