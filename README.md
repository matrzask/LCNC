# Shapley attribution in software engineering

The second-semester scope is in [TODO-Shapley-IO.md](TODO-Shapley-IO.md):
attribute prompt fragments to generated requirements, architecture components,
test cases, and agent actions or artifacts. The agent example evaluates all
eight subsets of three prompt fragments from the same initial repository state.

## Active files and references

| Path | Purpose |
| --- | --- |
| [TODO-Shapley-IO.md](TODO-Shapley-IO.md) | Professor's next steps, preserved unchanged. |
| [lcnc.py](lcnc.py) | Shared Ollama/JSON utilities, fragment IDs and generic exact or sampled Shapley attribution. |
| [prompt_to_feature.py](prompt_to_feature.py) | First-semester LC/NC generator and 18-feature evaluator, now isolated from the shared engine. |
| [requirements_engineering.py](requirements_engineering.py) | Requirement generation and outcome evaluator. |
| [architecture_design.py](architecture_design.py) | Component architecture generation and outcome evaluator. |
| [test_generation.py](test_generation.py) | Test-case generation and outcome evaluator. |
| [run_domain_experiment.py](run_domain_experiment.py) | CLI pilot runner for all four domains. |
| [visualize.py](visualize.py) | Attribution heatmaps, dependency graphs, importance and stability plots. |
| [projects/](projects/) | Professor-provided example projects, documentation, presentations and original submission archives. |
| [articles/](articles/) | Professor-provided article and presentation. |
| [old/](old/) | First-semester experiments, refinement UI, report source, original TODO and result snapshots. See its [archive notes](old/README.md). |

The shared engine takes a generator and an evaluator that returns numeric
outcome scores. It evaluates every coalition by default and supports sampling
for larger prompts. Run a pilot with `python run_domain_experiment.py
requirements`; choose `architecture`, `testing`, or `prompt-to-feature` for the
other domains. `--samples N` enables sampled estimation; omitting it enumerates
all coalitions. The first-semester implementation remains in
[old/lcnc.py](old/lcnc.py), and scripts under `old/` explicitly import both
`old.lcnc` and `old.visualize`.

The three new evaluators use deterministic keyword checks over generated
structured outputs. Treat their scores as pilot measures: they are easy to
inspect, but do not replace human validation of semantic correctness. The
agent-system domain in the TODO is not part of this milestone; it needs a runner
that restores the same repository state for every prompt coalition.

The generators use Python, `requests`, and Ollama at
`http://localhost:11434`; the default model is configured in [lcnc.py](lcnc.py).
Progress bars are optional (`tqdm`). Visualization uses `numpy`, `pandas`,
`matplotlib` and `networkx`. The local virtual environment is ignored by Git.

## Cleanup and version control

The cleanup preserves historical results because model calls and manual
refinement sessions cannot be reproduced exactly by simply rerunning a script.
Archived JSON, text and figures are eligible for version control, including
results previously hidden by broad ignore rules. Root-level demo outputs,
Python caches, the virtual environment and Windows metadata are ignored.

Commit the reorganization, both READMEs, the new TODO, professor-provided
materials, report source and archived experiment snapshots together. Git will
recognize unchanged tracked scripts and the HTML template as moves once both
their old and new paths are staged. No commit or staging is performed by the
cleanup itself.
