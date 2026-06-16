import json
from pathlib import Path
from statistics import mean

import numpy as np

from visualize import plot_fragment_feature_stability, plot_stability


def _average_ranks_desc(values_by_fragment: dict[str, float]) -> dict[str, float]:
    """Return average ranks (1 = highest value), handling ties."""
    ordered = sorted(values_by_fragment.items(), key=lambda x: x[1], reverse=True)
    ranks: dict[str, float] = {}

    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg_rank
        i = j + 1

    return ranks


def _spearman_from_rank_dicts(r1: dict[str, float], r2: dict[str, float]) -> float:
    fragments = sorted(set(r1.keys()) & set(r2.keys()))
    x = np.array([r1[f] for f in fragments], dtype=float)
    y = np.array([r2[f] for f in fragments], dtype=float)

    if len(x) < 2:
        return 1.0

    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0:
        return 1.0

    return float(np.corrcoef(x, y)[0, 1])


def _fragment_run_score(feat_map: dict[str, list[float]], run_idx: int) -> float:
    # Fragment score per run: sum of absolute attributions across all features.
    return float(sum(abs(scores[run_idx]) for scores in feat_map.values()))


def compute_stability_metrics(stability_scores: dict) -> dict:
    first_fragment = next(iter(stability_scores.values()))
    first_scores = next(iter(first_fragment.values()))
    num_runs = len(first_scores)

    run_rankings: list[dict[str, float]] = []
    fragment_rank_history: dict[str, list[float]] = {frag: [] for frag in stability_scores}

    for run_idx in range(num_runs):
        per_fragment_score = {
            frag: _fragment_run_score(feat_map, run_idx)
            for frag, feat_map in stability_scores.items()
        }
        run_ranking = _average_ranks_desc(per_fragment_score)
        run_rankings.append(run_ranking)

        for frag, rank in run_ranking.items():
            fragment_rank_history[frag].append(rank)

    # Pairwise Spearman agreement between run rankings.
    pairwise_spearman = []
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            pairwise_spearman.append(_spearman_from_rank_dicts(run_rankings[i], run_rankings[j]))

    mean_spearman = float(mean(pairwise_spearman)) if pairwise_spearman else 1.0

    changed_fragments = [
        frag for frag, ranks in fragment_rank_history.items() if len(set(ranks)) > 1
    ]
    ranking_stability_percent = 100.0 * (1.0 - (len(changed_fragments) / len(fragment_rank_history)))

    # Per-fragment stability: average of per-feature standard deviations across runs.
    fragment_feature_stds = {}
    for frag, feat_map in stability_scores.items():
        feature_stds = []
        for scores in feat_map.values():
            if len(scores) < 2:
                feature_stds.append(0.0)
            else:
                feature_stds.append(float(np.std(scores)))
        fragment_feature_stds[frag] = {
            "feature_stds": {
                feature: (float(np.std(scores)) if len(scores) > 1 else 0.0)
                for feature, scores in feat_map.items()
            },
            "avg_feature_std": float(np.mean(feature_stds)) if feature_stds else 0.0,
        }

    top_unstable = sorted(
        ((frag, data["avg_feature_std"]) for frag, data in fragment_feature_stds.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    mean_abs_std = float(np.mean([data["avg_feature_std"] for data in fragment_feature_stds.values()])) if fragment_feature_stds else 0.0
    least_stable_fragment = top_unstable[0][0] if top_unstable else None

    return {
        "num_runs": num_runs,
        "mean_pairwise_spearman": mean_spearman,
        "ranking_stability_percent": ranking_stability_percent,
        "changed_fragments": changed_fragments,
        "fragment_feature_stds": fragment_feature_stds,
        "top_unstable": top_unstable,
        "mean_abs_std": mean_abs_std,
        "least_stable_fragment": least_stable_fragment,
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "ex4_stability_scores.json"
    output_plot = base_dir / "ex4_stability.png"

    with input_path.open("r", encoding="utf-8") as f:
        stability_scores = json.load(f)

    metrics = compute_stability_metrics(stability_scores)

    fig = plot_stability(stability_scores)
    fig.savefig(output_plot, dpi=200, bbox_inches="tight")

    least_stable_fragment = metrics["least_stable_fragment"]
    if least_stable_fragment is not None:
        fragment_plot = plot_fragment_feature_stability(
            least_stable_fragment,
            stability_scores[least_stable_fragment],
        )
        fragment_plot_path = base_dir / "ex4_least_stable_fragment.png"
        fragment_plot.savefig(fragment_plot_path, dpi=200, bbox_inches="tight")

    print("=== Stability Metrics ===")
    print(f"Runs: {metrics['num_runs']}")
    print(f"Mean pairwise Spearman: {metrics['mean_pairwise_spearman']:.4f}")
    print(f"Ranking stability (% unchanged fragments): {metrics['ranking_stability_percent']:.2f}%")
    print(f"Mean std of |attribution| across fragments: {metrics['mean_abs_std']:.4f}")
    print("\nTop 5 unstable fragments (std of |attribution|):")
    for frag, std_val in metrics["top_unstable"]:
        print(f"- {frag}: {std_val:.4f}")

    if least_stable_fragment is not None:
        print(f"\nLeast stable fragment: {least_stable_fragment}")
        print(f"Saved per-feature stability graph to: {fragment_plot_path}")

    print(f"\nSaved stability graph to: {output_plot}")


if __name__ == "__main__":
    main()
