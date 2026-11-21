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

The root [lcnc.py](lcnc.py) now provides a domain-independent Shapley engine:
pass a generator and an evaluator for the output type. It enumerates every
coalition by default and supports sampled estimation for larger prompts.
Extracted fragments carry stable IDs. The first-semester implementation is
preserved separately in [old/lcnc.py](old/lcnc.py); scripts under `old/` import
it explicitly, while current code should import the root `lcnc.py`. The old
agent-system example still needs a runner that resets repository state between
coalition runs.

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
