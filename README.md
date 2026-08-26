# ATHENA — Verifier-Controlled Fact Finding for HIPAA Compliance

ATHENA answers a regulatory permissibility question ("is this disclosure of
protected health information permitted under HIPAA?") by keeping a symbolic
verifier — a machine-checked Datalog encoding of the HIPAA Privacy Rule, run
in [Souffl&eacute;](https://souffle-lang.github.io/) — in control of what
evidence gets collected, instead of asking a language model to extract
everything up front and trusting the extraction. The verifier evaluates the
current fact state and decides which predicate its proof still depends on; a
language-model oracle answers exactly that one question, `TRUE` / `FALSE` /
`UNKNOWN`; a genuine `UNKNOWN` is preserved as `UNRESOLVED` rather than
silently treated as `FALSE`.

This document is the practical "what do I run, with what inputs, what comes
out" reference for the two halves of the system: the Datalog formalization
(Souffl&eacute;) and the Python evaluation harness that drives it.

This repository contains only what is directly needed to compile the
formalization, drive it end-to-end, and reproduce the paper's reported
results. Paper drafts, LaTeX source, and internal planning/audit documents
are not part of this repository.

---

## 1. Repository layout

```
.
├── souffle/                    The formalization itself (Datalog/Souffle).
│   ├── hipaa.dl                    Entry point -- includes every module.
│   ├── hipaa_introspect.dl         Same rules, plus every intermediate
│   │                                relation marked .output (used for
│   │                                debugging a specific scenario's proof).
│   ├── hipaa_compiled.dl           Same rules, wrapped for use with real
│   │                                .facts/CSV input files instead of
│   │                                inline Datalog facts.
│   ├── hierarchy/
│   │   └── hipaa_hierarchies.dl    Principal/role/attribute/purpose
│   │                                hierarchies (subsumption, e.g. a doctor
│   │                                is a provider is a covered entity).
│   └── modules/                    One file per HIPAA section (§164.502,
│                                    §164.506, §164.508, ... §164.530) plus
│                                    shared types/macros/stubs.
│
├── experiments/
│   ├── predicate_census/           Python: the oracle, the proof-directed
│   │   ├── run_oracle_interface.py     search, the baselines, and the
│   │   ├── incremental_acquisition.py  metrics/analysis scripts. See §4.
│   │   ├── llm_oracle.py
│   │   ├── decision_criticality.py
│   │   ├── atomic_acquisition.py
│   │   ├── baseline_comparison.py
│   │   ├── refined_baseline.py
│   │   ├── proof_obligations.py
│   │   ├── signed_polarity.py
│   │   ├── find_negations.py
│   │   ├── build_census.py
│   │   ├── obligation_driven_acquisition.py
│   │   ├── failure_attribution.py
│   │   └── consolidated_metrics.py
│   └── baselines/
│       └── run_baselines.py        Shared helpers imported by the scripts
│                                    above (LLM calls, fact parsing, Souffle
│                                    invocation, batch extraction/verify).
│
├── data/                       Benchmarks and every result file the paper's
│                                tables and figures are drawn from. See §5.
│
├── requirements.txt
├── .env.example                 Copy to `.env` and fill in whichever API
│                                 keys you need (see §2).
└── README.md                    This file.
```

---

## 2. Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Install Souffl&eacute; itself (not a Python package):

```bash
# macOS
brew install souffle

# check
souffle --version   # this repository was last verified against 2.5
```

If you plan to query either hosted model, copy `.env.example` to `.env` and
fill in the keys you need:

```bash
cp .env.example .env
```

| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | `--provider anthropic` (Claude-Sonnet-5) |
| `OPENAI_API_KEY` | `--provider openai` (GPT-5.5) |

The five open-weight models are served through a local
[Ollama](https://ollama.com) server instead (`--provider ollama`, no key
needed): start it with `ollama serve` and pull each model first —
`ollama pull gemma2:2b`, `gemma3:4b`, `phi4`, `llama3.1:8b`, `qwen3:14b`.

---

## 3. Running the Datalog / Souffl&eacute; side

Everything in this section is Souffl&eacute; only — no Python, no LLM, no
API key. All commands assume your working directory is `souffle/`.

### Checking one scenario by hand

The fastest way to inspect the formalization directly, with your own facts:

```bash
cd souffle
cat > /tmp/scenario.dl <<'EOF'
#include "hipaa_introspect.dl"

activerole("hospital_a", "hospital").
activerole("dr_smith", "physician").
msg_contains("msg_1", "patient_x", "diagnosis").
disclosure_attempted("hospital_a", "dr_smith", "patient_x", "msg_1", "diagnosis", "treatment").
EOF

mkdir -p /tmp/scenario_out
souffle -D /tmp/scenario_out /tmp/scenario.dl

cat /tmp/scenario_out/is_disclosure_allowed.csv   # non-empty -> ALLOW
cat /tmp/scenario_out/is_disclosure_denied.csv    # non-empty -> DENY
```

`hipaa_introspect.dl` marks every intermediate relation `.output`, so
`/tmp/scenario_out/` also contains files like `permitted_by_164_502_a_1_i.csv`
— useful for tracing exactly which rule fired. Use plain `hipaa.dl` instead
of `hipaa_introspect.dl` if you only want the top-level verdict and a
smaller output directory.

The six canonical arguments every disclosure-scoped relation shares are
`p1` (sender), `p2` (receiver), `q` (subject of the information), `m`
(message id), `t` (attribute/information type), `u` (purpose) — see
`souffle/modules/hipaa_types.dl`'s `#define ARGS` / `#define DECL_ARGS`.

---

## 4. Running the Python side

Every script below is invoked **from the repository root** as
`python3 experiments/predicate_census/<script>.py [flags]`. This matters for
two reasons: each script's internal `ROOT` is computed as two directories up
from its own file, so a `--corpus` value must be given exactly as
`data/athena_bench_...json` (repo-root-relative — not re-relativized to your
shell's current directory); and Python automatically puts each script's own
directory on `sys.path`, so its sibling-module imports (e.g.
`run_oracle_interface.py` importing `llm_oracle`) resolve correctly no matter
where you launch it from. Running everything from the repo root satisfies
both at once.

### 4.1 One-time setup scripts

These re-derive static analysis artifacts directly from the current
`souffle/**/*.dl` source. Their outputs are already committed in `data/`
(`predicate_census_edb.json`, `predicate_census_idb.json`,
`negation_audit.json`, `signed_polarity.json`, `gold30_proof_obligations.json`)
— **you do not need to re-run these** unless you've edited the formalization
and want the derived predicate census / negation audit / polarity
classification / proof-obligation counts to reflect your change.

```bash
python3 experiments/predicate_census/build_census.py       # -> data/predicate_census_edb.json, _idb.json
python3 experiments/predicate_census/find_negations.py     # -> data/negation_audit.json
python3 experiments/predicate_census/signed_polarity.py    # -> data/signed_polarity.json (enabler/blocker/
                                                             #    mixed classification the search relies on)
python3 experiments/predicate_census/proof_obligations.py  # -> data/gold30_proof_obligations.json (per-
                                                             #    scenario minimum-question counts, read by
                                                             #    the pipeline below for "optimal questions")
```

### 4.2 Main pipeline: verifier-controlled fact finding (ATHENA)

The core result of the paper. For one model, on one benchmark:

```bash
python3 experiments/predicate_census/run_oracle_interface.py \
    --model gemma3:4b --provider ollama \
    --corpus data/athena_bench_tier2_132.json
```

| Flag | Meaning | Default |
|---|---|---|
| `--model` | Model name for the given provider (see §2 for the exact strings used in this paper). | required |
| `--provider {ollama,anthropic,openai}` | Which backend serves `--model`. | required |
| `--corpus` | Benchmark JSON to evaluate against — `data/athena_bench_gold30.json`, `data/athena_bench_tier2_132.json`, or `data/athena_bench_adversarial.json`. | `data/athena_bench_gold30.json` |

Output: `data/<corpus-stem>_oracle_interface_<model>.json` — one record per
scenario with the full oracle-query trace, final status
(`PROVED_PERMITTED` / `PROVED_DENIED` / `UNRESOLVED`), and the independent
post-hoc soundness re-check (`dangerous_false_proof`). This is the file
every downstream metrics/baseline/analysis script reads.

Every `(model, provider)` pair used in the paper:

| Model | Provider |
|---|---|
| `gemma2:2b`, `gemma3:4b`, `phi4`, `llama3.1:8b`, `qwen3:14b` | `ollama` |
| `claude-sonnet-5` | `anthropic` |
| `gpt-5.5` | `openai` |

### 4.3 Baselines: Direct LLM, ETV-closed, ETV-open

The comparison architectures — one LLM call producing a direct verdict, and
batch extract-then-verify under closed- and open-world semantics:

```bash
python3 experiments/predicate_census/baseline_comparison.py \
    --model gemma3:4b --provider ollama \
    --corpus data/athena_bench_tier2_132.json
```

Same `--model` / `--provider` / `--corpus` flags as §4.2. Output:
`data/baseline_comparison_<corpus-stem>_<model>.json`, containing all three
baseline verdicts (`baseline_a` = Direct LLM, `baseline_b_closed` =
ETV-closed, `baseline_b_open` = ETV-open) per scenario, alongside the
matching ATHENA verdict for the same scenario/model for direct comparison.

### 4.4 Refined ETV (self-correction pass)

Re-shows the model its own ETV-closed extraction and asks it to correct
before re-verifying — run **after** §4.3 for the same model/corpus (it reads
that output):

```bash
python3 experiments/predicate_census/refined_baseline.py \
    --model gemma3:4b --provider ollama \
    --corpus data/athena_bench_tier2_132.json
```

Output: `data/refined_baseline_<corpus-stem>_<model>.json`.

### 4.5 Metrics and analysis (no new LLM calls)

Both scripts below read only already-computed `data/*_oracle_interface_*.json`
files — safe to re-run at any time, free, no API/Ollama calls.

```bash
# Full metrics table (accuracy, balanced accuracy, false-permit/false-deny
# rate, coverage, per-class recall) across every model with both corpora present:
python3 experiments/predicate_census/consolidated_metrics.py --corpus both --format table

# Root-cause trace of every false-deny case for a given model (or "all"):
python3 experiments/predicate_census/failure_attribution.py --model gemma3_4b
```

`consolidated_metrics.py --format` also accepts `markdown` or `json`.
`failure_attribution.py` is currently wired to `data/gold30_oracle_interface_*.json`
only; pass `--model all` for every model with a gold30 trace present, or edit
its `MODELS`/corpus list at the top of the file to point at `tier2_132`
instead if you want the appendix's exact "8 cases across both benchmarks"
figure reproduced against a different corpus.

### 4.6 Search-cost trace (no LLM)

`incremental_acquisition.py` runs the same proof-obligation-driven search
`run_oracle_interface.py` (§4.2) uses, but standalone and Souffl&eacute;-only
— on gold30 it substitutes a perfect ground-truth lookup for the oracle
(no LLM call, no API/Ollama needed); on `tier2_132`/`adversarial`, which have
no independent per-fact ground truth, every query returns `UNKNOWN` and the
run measures search cost only (verifier-evaluation count per scenario), not
correctness:

```bash
python3 experiments/predicate_census/incremental_acquisition.py --corpus data/athena_bench_gold30.json
```

Output: `data/<corpus-stem>_incremental_traces.json` (`gold30` for the
default corpus). This is what backs the search-cost claims in the paper's
evaluation-cost discussion.

### 4.7 Validation-only reference (not part of the main pipeline)

`obligation_driven_acquisition.py` is the exhaustive, small-instance
reference search implementation §4.6's `incremental_acquisition.py` (the
real, scaled-up search used throughout the pipeline) was checked against for
correctness. It is not invoked by anything else and produces
`data/gold30_obligation_driven_traces.json` — the file
`incremental_acquisition.py`'s own docstring says must match exactly before
any change to the live search algorithm is trusted:

```bash
python3 experiments/predicate_census/obligation_driven_acquisition.py
```

---

## 5. Data directory reference

| File(s) | What it is |
|---|---|
| `athena_bench_gold30.json` | 30-scenario benchmark with independent, hand-verified predicate-level ground truth. Used to validate the search algorithm and establish the oracle ceiling — not a source of headline results. |
| `athena_bench_tier2_132.json` | **Primary evaluation benchmark** (GoldCoin-HHS): 132 real-world HIPAA disclosure scenarios (103 court cases, 29 HHS case examples). Scenario-level ground truth only. |
| `athena_bench_adversarial.json` | 50-scenario narrative-robustness benchmark derived from GoldCoin-HHS (rhetorical rewording / evidence injection / evidence omission). |
| `gold30_independent_ground_truth.json` | Per-predicate, per-scenario TRUE/FALSE/UNKNOWN annotation for gold30, independent of any model. |
| `gold30_proof_obligations.json` | Per-scenario minimum-question counts for gold30 (§4.1 output). |
| `signed_polarity.json`, `negation_audit.json`, `predicate_census_edb.json`, `predicate_census_idb.json` | Static analysis of the formalization itself (§4.1 outputs) — predicate polarity classification, negation structure, derived-vs-oracle-facing census. |
| `{corpus}_oracle_interface_{model}.json` | ATHENA's full per-scenario trace for one (corpus, model) pair (§4.2 output). `{corpus}` is `gold30`, `tier2_132`, or `adversarial`. |
| `baseline_comparison_{corpus}_{model}.json` | Direct LLM / ETV-closed / ETV-open verdicts for one (corpus, model) pair (§4.3 output). |
| `refined_baseline_{corpus}_{model}.json` | Refined-ETV verdicts for one (corpus, model) pair (§4.4 output). |
| `{corpus}_incremental_traces.json` | Model-independent proof-search trace (verifier-evaluation count per scenario) for one corpus — a perfect ground-truth oracle on gold30, search-cost-only elsewhere (§4.6 output). Backs the search-cost claims in the paper's evaluation-cost discussion. |
| `consolidated_metrics.json`, `failure_attribution.json` | Cached output of the §4.5 analysis scripts. |

Every `{model}` slot above uses the lowercased, punctuation-to-underscore
form of the CLI `--model` string (e.g. `--model gemma3:4b` &rarr;
`gemma3_4b`, `--model claude-sonnet-5` &rarr; `claude_sonnet_5`).

---

## 6. Reproducing the paper's reported results

The headline results (RQ1&ndash;RQ4, all seven models) come from running
§4.2, §4.3, and §4.4 for every `(model, provider)` pair listed in §4.2 with
`--corpus data/athena_bench_tier2_132.json`, then §4.5's
`consolidated_metrics.py`. The narrative-robustness appendix result comes
from the same three steps with `--corpus data/athena_bench_adversarial.json`.
All of these output files are already committed in `data/` (§5) — you do not
need to re-run any live LLM calls to inspect or recompute the reported
metrics from them; you only need to re-run §4.2&ndash;§4.4 if you want to
regenerate a trace from scratch (e.g. against a different model).
