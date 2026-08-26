"""
The LLM oracle -- answers exactly the predicate-level question the
verifier's search selects, TRUE/FALSE/UNKNOWN, in place of a perfect
lookup oracle.

Predicates fall into three phrasing classes, chosen per-predicate rather
than from one uniform template:
  - lacks_authorized_relationship, inconsistent_with_notice_of_privacy_practices:
    negatively framed in the formalization -- ask the model the POSITIVE
    counterpart question and negate the answer symbolically. Asking the
    negative form directly was found empirically to produce a much higher
    unsupported-TRUE rate on these two guards (roughly 40%, vs. roughly 2%
    when the positive counterpart is asked and negated instead).
  - ce_wrongly_denied_access: asking directly, in either polarity, leaks an
    evaluative legal conclusion into the question itself, so no safe
    direct/reversed phrasing exists. A direct TRUE/FALSE/UNKNOWN judgment
    call is replaced by atomic evidence extraction
    (atomic_acquisition.py's CWD_FIELDS_F /
    derive_ce_wrongly_denied_f), accepting some loss of retention accuracy
    in exchange for not over-committing. This is the only predicate in this
    module using the atomic-extraction path instead of a single
    direct/reversed question.
  - obtained_authorization_164_508, is_required_by_law, believes_minimum_necessary,
    consistent_with_applicable_law: already positively framed -- asked
    directly, no reversal needed.

Question framing otherwise asks explicitly whether the narrative's EVIDENCE
establishes the fact, not whether it is plausible -- the phrasing found to
score best on positive-predicate accuracy during prompt-phrasing
development.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import run_baselines as base  # noqa: E402
from atomic_acquisition import (  # noqa: E402
    CWD_FIELDS_F, derive_ce_wrongly_denied_f, _atomic_extract,
)

REVERSED_TRUE_COMPLEMENT = {
    "lacks_authorized_relationship":
        lambda a: f"Does {a[1]} have a legitimate treatment or operational relationship to {a[2]}, the subject of this information?",
    "inconsistent_with_notice_of_privacy_practices":
        lambda a: "Is this disclosure consistent with the covered entity's posted notice of privacy practices?",
}

def _humanize_entity(entity):
    """snake_case/kebab-case -> Title Case, e.g. saint_francis_hospital ->
    'Saint Francis Hospital', so the question text can string-match the
    narrative's own proper nouns instead of a raw internal identifier."""
    return entity.replace("_", " ").replace("-", " ").title()


# Role values in this formalization sometimes stand for a broader legal
# category than the plain English word suggests. "hospital" is used as a
# generic covered-entity/health-record-holder tag, not literally "is this
# a hospital building" -- narratives describing a pain-management clinic,
# an employer holding occupational-health records, or a mental-health
# service don't use the word "hospital," so a literally correct "no" to
# "is this a hospital?" would produce the wrong Datalog fact. Scoped to
# "hospital" only -- the one role value with evidenced failures; other
# role values are asked at face value until similarly evidenced.
ROLE_BROADENING = {
    "hospital": ("a HIPAA-covered healthcare provider or organization that creates, "
                 "holds, or maintains health records -- this includes hospitals, "
                 "clinics, medical practices, pharmacies, and employers who maintain "
                 "occupational health records"),
}


def _article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def _role_scope(role):
    return ROLE_BROADENING.get(role, f"{_article(role)} {role}")


def _activerole_phrase(a):
    entity, role = a[0], a[1]
    return f"In this scenario, does the narrative establish that {_humanize_entity(entity)} functions as {_role_scope(role)}?"


def _belongstorole_phrase(a):
    entity, role = a[0], a[1]
    return f"Does the narrative establish that {_humanize_entity(entity)} is or has the status of {_role_scope(role)}?"


DIRECT_PHRASE = {
    "obtained_authorization_164_508":
        lambda a: "Did the covered entity obtain a valid written authorization from the individual for this disclosure?",
    "is_required_by_law":
        lambda a: "Is this disclosure required by law -- e.g. mandated by a statute or regulation, not merely permitted?",
    "believes_minimum_necessary":
        lambda a: "Was the information disclosed limited to what was reasonably necessary for the stated purpose?",
    "consistent_with_applicable_law":
        lambda a: "Is this disclosure consistent with (not prohibited by) other applicable law?",
    "activerole": _activerole_phrase,
    "belongstorole": _belongstorole_phrase,
}


def _generic_phrase(name, args):
    return f"Does the narrative establish that {name.replace('_', ' ')} holds for ({', '.join(args)})?"


def _prompt(narrative, question):
    return f"""You are answering one narrow factual question about a narrative, to feed a formal legal verifier. You are NOT making a legal judgment -- only reporting whether the EVIDENCE in the narrative establishes this one fact.

NARRATIVE:
{narrative}

QUESTION: Based only on the scenario text, is there sufficient evidence that: {question}

Answer:
TRUE -- the scenario establishes this.
FALSE -- the scenario establishes that this is NOT the case.
UNKNOWN -- the scenario does not establish either way.

Do not treat "plausible" or "not contradicted" as the same as "established." Absence of any mention is UNKNOWN, not FALSE and not TRUE.

Respond in exactly this format, nothing else:
VALUE: <TRUE|FALSE|UNKNOWN>
"""


def _parse(text):
    m = re.search(r'VALUE:\s*(TRUE|FALSE|UNKNOWN)', text, re.I)
    return m.group(1).upper() if m else "UNKNOWN"


def _negate(val):
    return {"TRUE": "FALSE", "FALSE": "TRUE", "UNKNOWN": "UNKNOWN"}[val]


# ---------- believes_minimum_necessary decomposition. is_required_by_law has
# no equivalent validated decomposition and is deliberately left unchanged --
# do not invent one here. ----------

def _prompt_c(narrative, question):
    return f"""You are answering one narrow factual question about a narrative, to feed a formal legal verifier.

NARRATIVE:
{narrative}

QUESTION: {question}

Rules:
- First quote the exact span(s) of the narrative that establish the answer.
- If no such span exists, EVIDENCE must be NONE.
- If EVIDENCE is NONE, VALUE must be UNKNOWN -- there is no exception to this rule.

Respond in exactly this format, nothing else:
EVIDENCE: <quoted span(s), or NONE>
VALUE: <TRUE|FALSE|UNKNOWN>
"""


def _parse_evidence(text):
    m = re.search(r'EVIDENCE:\s*(.*?)(?:\n\s*VALUE:|\Z)', text, re.I | re.S)
    if not m:
        return None
    span = m.group(1).strip()
    if not span or span.upper().startswith("NONE"):
        return None
    return span


def _ask_subq(narrative, question):
    resp = base.ollama(_prompt_c(narrative, question))
    evidence = _parse_evidence(resp)
    value = _parse(resp)
    if evidence is None:
        value = "UNKNOWN"
    return value, evidence


def _extract_span(narrative, question, none_token, max_frac=1.0):
    prompt = f"""You are extracting text from a narrative for a formal legal verifier. You are NOT making any judgment -- only quoting text, or reporting that no such text exists.

NARRATIVE:
{narrative}

TASK: {question}

Respond with exactly the quoted span, or exactly the word {none_token} if no such span exists. Nothing else.
"""
    resp = base.ollama(prompt).strip()
    cleaned = resp.strip('"“”\' \n')
    if not cleaned or cleaned.upper() == none_token or none_token in resp.upper().split():
        return None
    if max_frac < 1.0 and narrative and len(cleaned) > max_frac * len(narrative):
        return None
    return resp


def _extract_narrowing(narrative):
    q_disclosed = ("Quote the exact SHORT span (a phrase, at most one sentence) of the scenario that lists or "
                   "describes the SPECIFIC items of information that were disclosed or accessed. If the scenario "
                   "does not specify particular items, respond exactly: NOT_SPECIFIED")
    q_subset = ("Quote the exact SHORT span (a phrase, at most one sentence -- NOT a paragraph, NOT the whole "
               "scenario) of the scenario, if any, indicating the disclosed items were a SUBSET of a larger set of "
               "information -- for example: items explicitly described as NOT included; language such as 'only', "
               "'limited to', 'excluding', 'basic', 'specific', or 'particular'; or a direct contrast with 'the "
               "complete/entire/full record'. A sentence that merely lists what was disclosed, without any such "
               "contrasting or limiting language, is NOT a match. If no short span meeting this description exists, "
               "respond exactly: NONE")
    q_complete = ("Quote the exact SHORT span (a phrase, at most one sentence -- NOT a paragraph, NOT the whole "
                 "scenario) of the scenario, if any, stating that the COMPLETE, ENTIRE, or FULL record or file was "
                 "disclosed, with nothing withheld. If no short span meeting this description exists, respond "
                 "exactly: NONE")
    disclosed = _extract_span(narrative, q_disclosed, none_token="NOT_SPECIFIED")
    subset = _extract_span(narrative, q_subset, none_token="NONE", max_frac=0.35)
    complete = _extract_span(narrative, q_complete, none_token="NONE", max_frac=0.35)
    if disclosed is None:
        return "UNKNOWN"
    if subset is not None:
        return "TRUE"
    if complete is not None:
        return "FALSE"
    return "UNKNOWN"


def decompose_believes_minimum_necessary(narrative, args):
    q_purpose = ("Does the scenario establish a legitimate treatment, payment, healthcare-operations, or other "
                 "stated purpose for this specific disclosure -- or does it instead show the discloser had no such "
                 "purpose (e.g. curiosity, no assigned role, no treatment relationship)?")
    q_scope = ("Does the scenario describe the specific information items actually disclosed (naming or describing "
               "particular data elements), rather than only referring to 'the records' or 'the file' as an "
               "undifferentiated whole?")
    v_purpose, _ = _ask_subq(narrative, q_purpose)
    if v_purpose == "FALSE":
        return "FALSE"
    if v_purpose == "UNKNOWN":
        return "UNKNOWN"
    v_scope, _ = _ask_subq(narrative, q_scope)
    if v_scope != "TRUE":
        return "UNKNOWN"
    return _extract_narrowing(narrative)


def make_llm_oracle(narrative, model, provider):
    """Returns oracle(name, args) -> "TRUE"|"FALSE"|"UNKNOWN", calling `model` live."""
    base.MODEL, base.PROVIDER = model, provider

    def oracle(name, args):
        if name == "ce_wrongly_denied_access":
            evidence = _atomic_extract(narrative, CWD_FIELDS_F)
            return derive_ce_wrongly_denied_f(evidence)
        # believes_minimum_necessary: uses the single-question direct phrasing
        # here because this predicate is oracle-facing by design -- HIPAA
        # leaves "minimum necessary" to subjective/professional judgment, not
        # something computed from the structure of the rules themselves. The
        # decomposition
        # (purpose_exists/scope_described/narrowing_observed, derived in
        # Python) is kept below as `decompose_believes_minimum_necessary` for
        # the paper's ablation only: it does not generalize across models, so
        # it is not routed here.
        if name in REVERSED_TRUE_COMPLEMENT:
            question = REVERSED_TRUE_COMPLEMENT[name](args)
            raw = _parse(base.ollama(_prompt(narrative, question)))
            return _negate(raw)
        question = DIRECT_PHRASE.get(name, lambda a: _generic_phrase(name, a))(args)
        return _parse(base.ollama(_prompt(narrative, question)))

    return oracle
