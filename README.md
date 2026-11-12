# Shapley attribution in software engineering

The second-semester scope is in [TODO-Shapley-IO.md](TODO-Shapley-IO.md):
attribute prompt fragments to generated requirements, architecture components,
test cases, and agent actions or artifacts. The agent example evaluates all
eight subsets of three prompt fragments from the same initial repository state.

## Active files and references

| Path | Purpose |
| --- | --- |
| [TODO-Shapley-IO.md](TODO-Shapley-IO.md) | Professor's next steps, preserved unchanged. |
| [lcnc.py](lcnc.py) | Reusable starting point: Ollama calls, JSON parsing, prompt decomposition, feature differences, leave-one-out and sampled attribution. |
| [visualize.py](visualize.py) | Reusable attribution heatmaps, dependency graphs, importance and stability plots. |
| [projects/](projects/) | Professor-provided example projects, documentation, presentations and original submission archives. |
| [articles/](articles/) | Professor-provided article and presentation. |
| [old/](old/) | First-semester experiments, refinement UI, report source, original TODO and result snapshots. See its [archive notes](old/README.md). |

The retained code is still the first-semester implementation. Its generator and
18-feature extractor describe LC/NC applications; they need domain-specific
generators and evaluators for the new tasks. The sampled attribution function
also calls that extractor directly, so it is a starting point rather than a
domain-independent engine. It does not yet implement exact enumeration of all
coalitions or resetting an agent's repository state between runs.

Before extending the estimator, review its treatment of failed generations
(skipped samples still count in the denominator), fragment identity (text is
used as the key), and coalition ordering (sampled fragments are shuffled and the
added fragment is appended). These affect interpretation of the new experiments.

The core uses Python and `requests`, with optional `tqdm`; visualization uses
`numpy`, `pandas`, `matplotlib` and `networkx`. The generator expects Ollama at
`http://localhost:11434` with the `mistral` model. The local virtual environment
is ignored by Git.

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
