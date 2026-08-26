"""
Predicate census, built directly from the current .dl source.

Extracts every .decl across souffle/modules/*.dl and souffle/hierarchy/*.dl,
classifies EDB vs. IDB (a relation is IDB iff a `:-` rule body appears
anywhere in the source), builds the predicate
dependency graph, and computes reachability from the four decision-output relations
in hipaa_top.dl (is_disclosure_allowed, is_disclosure_denied,
formal_access_request_must_be_granted, formal_access_request_properly_denied).

Requires a proper paren-depth/string-aware statement parser, not line-based regex:
several .decl statements and rule bodies in hipaa_164_512.dl span multiple lines,
which a naive per-line parser silently drops.

Usage: python3 build_census.py
Writes: ../../data/predicate_census_edb.json, ../../data/predicate_census_idb.json
"""
import glob
import json
import re
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = list((ROOT / "souffle/modules").glob("*.dl")) + list((ROOT / "souffle/hierarchy").glob("*.dl"))

DECISION_ROOTS = [
    "is_disclosure_allowed", "is_disclosure_denied",
    "formal_access_request_must_be_granted", "formal_access_request_properly_denied",
]

NEG_PATTERNS = [
    r'^lacks_', r'^violates_', r'^inconsistent_', r'^wrongly_', r'^ce_wrongly_',
    r'non_requested', r'^prohibited_', r'unlawful',
    r'^charged_impermissible', r'^charged_unreasonable', r'^lacks_reasonable',
    r'jeopardizes_', r'^require_authorization', r'defective_authorization',
]

BUILTINS = {"match", "contains", "strlen", "substr", "cat", "ord", "to_string",
            "to_number", "count", "sum", "min", "max", "range"}


def strip_comments(text):
    return "\n".join(line[:line.find("//")] if "//" in line else line for line in text.split("\n"))


def remove_directives(text):
    """Strip .decl/.output/.input/... statements, including ones spanning multiple lines."""
    lines = text.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        line = lines[i]
        if re.match(r'^\s*\.\w', line):
            depth = line.count("(") - line.count(")")
            i += 1
            while depth > 0 and i < n:
                depth += lines[i].count("(") - lines[i].count(")")
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def split_clauses(body_text):
    """Split into statements terminated by a '.' at paren-depth 0, respecting string literals."""
    clauses, depth, in_str, buf = [], 0, False, []
    for i, c in enumerate(body_text):
        buf.append(c)
        if in_str:
            if c == '"' and body_text[i - 1] != '\\':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == '.' and depth == 0:
            clause = "".join(buf).strip()
            if clause and clause != ".":
                clauses.append(clause)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        clauses.append(tail)
    return clauses


def extract_decls(files):
    decls, arity_of = {}, {}
    decl_re = re.compile(r'^\.decl\s+([a-zA-Z_][a-zA-Z0-9_]*)\(', re.M)
    for fn in files:
        text = fn.read_text()
        for m in decl_re.finditer(text):
            name = m.group(1)
            decls[name] = str(fn.relative_to(ROOT))
            start, depth, i = m.end(), 1, m.end()
            while depth > 0 and i < len(text):
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                i += 1
            argtext = text[start:i - 1]
            arity_of[name] = len([a for a in argtext.split(",") if a.strip()])
    return decls, arity_of


def extract_rules(files, decls):
    rules = defaultdict(list)
    for fn in files:
        text = remove_directives(strip_comments(fn.read_text()))
        for clause in split_clauses(text):
            hm = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', clause)
            if not hm or hm.group(1) not in decls or ":-" not in clause:
                continue
            head = hm.group(1)
            body = clause.split(":-", 1)[1]
            body_preds = set(re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', body)) - BUILTINS
            body_preds.discard(head)
            body_preds &= set(decls.keys())
            rules[head].append((str(fn.relative_to(ROOT)), body_preds))
    return rules


def polarity(name):
    return "negative" if any(re.search(p, name) for p in NEG_PATTERNS) else "positive"


def main():
    decls, arity_of = extract_decls(FILES)
    rules = extract_rules(FILES, decls)
    idb_names = set(rules.keys())
    edb_names = set(decls.keys()) - idb_names

    body_of = defaultdict(set)
    for head, clauselist in rules.items():
        for _fn, preds in clauselist:
            body_of[head] |= preds

    reachable, q = set(), deque(DECISION_ROOTS)
    while q:
        x = q.popleft()
        if x in reachable:
            continue
        reachable.add(x)
        q.extend(c for c in body_of.get(x, set()) if c not in reachable)

    parents = defaultdict(set)
    for head, preds in body_of.items():
        for p in preds:
            parents[p].add(head)

    gt_path = ROOT / "data/gold30_independent_ground_truth.json"
    gt_by_pred = defaultdict(lambda: {"TRUE": 0, "FALSE": 0, "UNKNOWN": 0})
    if gt_path.exists():
        for row in json.loads(gt_path.read_text()):
            gt_by_pred[row["predicate"]][row["independent_value"]] += 1

    def module_of(fn):
        return fn.split("/")[-1].replace(".dl", "")

    edb_rows = []
    for name in sorted(reachable & edb_names):
        gtc = gt_by_pred.get(name)
        cov = f"T{gtc['TRUE']}/F{gtc['FALSE']}/U{gtc['UNKNOWN']}" if gtc else "not in gold30 candidate lists"
        edb_rows.append({
            "predicate": name, "arity": arity_of[name], "module": module_of(decls[name]),
            "polarity": polarity(name), "used_by": sorted(parents.get(name, [])),
            "gold30_coverage": cov,
        })

    idb_rows = []
    for name in sorted(reachable & idb_names):
        idb_rows.append({
            "predicate": name, "arity": arity_of[name], "module": module_of(decls[name]),
            "used_by": sorted(parents.get(name, [])),
        })

    unreachable = sorted(set(decls.keys()) - reachable)

    out_dir = ROOT / "data"
    (out_dir / "predicate_census_edb.json").write_text(json.dumps(edb_rows, indent=2))
    (out_dir / "predicate_census_idb.json").write_text(json.dumps(idb_rows, indent=2))

    print(f"Total declared relations: {len(decls)}")
    print(f"Reachable from decision outputs: {len(reachable)}")
    print(f"  reachable EDB (candidate pool): {len(edb_rows)}")
    print(f"  reachable IDB (Datalog-internal): {len(idb_rows)}")
    print(f"Unreachable (dead code): {len(unreachable)} -> {unreachable}")
    covered = sum(1 for r in edb_rows if "not in gold30" not in r["gold30_coverage"])
    print(f"EDB predicates with gold30 coverage: {covered}/{len(edb_rows)}")


if __name__ == "__main__":
    main()
