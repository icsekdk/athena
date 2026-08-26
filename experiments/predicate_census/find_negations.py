"""
Negation-occurrence census, built on top of build_census.py's parser.

Finds every `!predicate(...)` occurrence inside a rule body anywhere in the
reachable decision graph (souffle/modules/*.dl, souffle/hierarchy/*.dl), and
records which head predicate each negation occurs in. This generalizes the
manual finding that the 9 top-level guards are negated directly inside
is_disclosure_allowed to the whole graph: negation-as-failure over EDB and
IDB predicates alike happens at every level of the proof tree, not just the
top-level conjunction.

Usage: python3 find_negations.py
Writes: ../../data/negation_audit.json
"""
import json
import re
from pathlib import Path

from build_census import (
    ROOT, FILES, extract_decls, extract_rules, strip_comments, remove_directives,
    split_clauses,
)

NEG_RE = re.compile(r'!\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(')


def main():
    decls, arity_of = extract_decls(FILES)
    rules = extract_rules(FILES, decls)
    idb_names = set(rules.keys())
    edb_names = set(decls.keys()) - idb_names

    negations = {}  # negated_pred -> sorted list of head predicates that negate it
    for fn in FILES:
        text = remove_directives(strip_comments(fn.read_text()))
        for clause in split_clauses(text):
            hm = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', clause)
            if not hm or hm.group(1) not in decls or ":-" not in clause:
                continue
            head = hm.group(1)
            body = clause.split(":-", 1)[1]
            for m in NEG_RE.finditer(body):
                negated = m.group(1)
                if negated not in decls:
                    continue
                negations.setdefault(negated, set()).add(head)

    rows = []
    for pred, heads in sorted(negations.items()):
        rows.append({
            "predicate": pred,
            "type": "EDB" if pred in edb_names else "IDB",
            "negated_in": sorted(heads),
        })

    edb_rows = [r for r in rows if r["type"] == "EDB"]
    idb_rows = [r for r in rows if r["type"] == "IDB"]

    out_path = ROOT / "data/negation_audit.json"
    out_path.write_text(json.dumps(rows, indent=2))

    print(f"Total distinct negated predicates: {len(rows)}")
    print(f"  negated EDB (real guard candidates): {len(edb_rows)}")
    print(f"  negated IDB (must stay Datalog-internal, never oracle-facing): {len(idb_rows)}")


if __name__ == "__main__":
    main()
