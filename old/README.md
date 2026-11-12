# First-semester archive

These files record the LC/NC prompt-to-feature project. Reusable code remains
in [../lcnc.py](../lcnc.py) and [../visualize.py](../visualize.py); the current
assignment is [../TODO-Shapley-IO.md](../TODO-Shapley-IO.md).

## Contents

| Files | Why retained |
| --- | --- |
| `ex1_feature_drop_rate.py`, `ex1/` | Top-versus-random fragment removal experiment, fragment rankings and measured results. |
| `ex2_refinement.py`, `ex2_web_app.py`, `templates/`, `ex2/` | CLI and Flask refinement study, UI, and manually annotated prompt histories. |
| `ex3_cost_vs_quality.py`, `ex3/` | Sampling cost, stability, feature-drop metrics and associated figures. |
| `ex4_stability.py`, `ex4_analyze.py`, `ex4_*` | Repeated-run stability experiment, raw scores and analysis figures. |
| `attributions.json`, `stability*`, root-level `ex3_*.json` and `ex3_*.png`, `plots/` | Additional historical runs and demonstration figures; not assumed interchangeable with the final experiment results. |
| `raport.tex`, `todo.txt` | Report source and original scope/research questions. The submitted PDF is [../projects/B/LCNC.pdf](../projects/B/LCNC.pdf). |
| `ex3.zip` | Earlier experiment snapshot: three files differ from `ex3/`, so this archive is not a disposable duplicate. |
| `materialy/` | Unique older article excerpt. The duplicate reference implementation and PDF remain in [../projects/A/](../projects/A/). |

The `ex4/` directory was empty; the existing scripts read and write `ex4_*`
files beside the scripts instead. Empty directories are not stored by Git.

The refinement histories contain handwritten `//` comments. They are preserved
as study records and are not strict JSON. The report source already referred to
two absent figures, `ex2_webui.png` and `ex4_least_stable.png`; the cleanup does
not repair or invent those assets. Compile it from this directory to resolve
the existing relative `plots/` and `ex3/` references.

## Running archived scripts

Use this directory as the working directory and add the repository root to
`PYTHONPATH`, so existing imports and relative output paths keep working. From
the repository root in PowerShell, for example:

```powershell
$previousPythonPath = $env:PYTHONPATH
$repoRoot = (Get-Location).Path
$env:PYTHONPATH = $repoRoot + [IO.Path]::PathSeparator + $previousPythonPath
Push-Location old
try {
    & ../.venv/Scripts/python.exe ex2_web_app.py
} finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
```

Other experiment scripts can be invoked the same way. The UI additionally
requires `flask`, `flask-cors` and `plotly`. Generating new results requires
Ollama and may overwrite archived output files; Git can show those changes.

## Verified duplicate removal

- Root `LCNC.pdf` and `LCNC.zip` were byte-identical to the copies in
  [../projects/B/](../projects/B/), which remain untouched.
- Root `kod.zip` and `wyniki.zip` were byte-identical to entries in the retained
  [../projects/B/LCNC.zip](../projects/B/LCNC.zip). Every code/result file in those
  ZIPs also matched its loose counterpart, including the experiment-four files.
- `materialy/SP2_lowcodenocode.py` and its accompanying documentation PDF matched
  the copies in [../projects/A/](../projects/A/).
- `plots/ex4_stability_graph2.png` matched `ex4_stability.png` here.
- The root Python bytecode cache was removed. Its old contents were generated
  files, and future caches are ignored.

All duplicate comparisons used SHA-256 of file contents, including decompressed
ZIP entries. Other result snapshots and professor-provided archives were kept.
