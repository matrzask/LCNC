import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict

import requests

from lcnc import extract_prompt_fragments, generate_architecture, shap_sampling_attribution


DEFAULT_PROMPT = """
The system should allow users to register and log in.
It must also provide a dashboard for managing their profiles and settings.
Additionally, the application should support offline access and send notifications for important updates.
""".strip()


def _rankdata(values: list[float]) -> list[float]:
	"""Return 1-based average ranks (handles ties)."""
	indexed = list(enumerate(values))
	indexed.sort(key=lambda t: t[1])
	ranks = [0.0] * len(values)
	i = 0
	while i < len(indexed):
		j = i
		while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
			j += 1
		avg_rank = (i + 1 + j + 1) / 2.0
		for k in range(i, j + 1):
			ranks[indexed[k][0]] = avg_rank
		i = j + 1
	return ranks


def _pearson_corr(x: list[float], y: list[float]) -> float:
	if len(x) != len(y) or len(x) < 2:
		return float("nan")
	x_mean = sum(x) / len(x)
	y_mean = sum(y) / len(y)
	num = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
	dx = sum((a - x_mean) ** 2 for a in x)
	dy = sum((b - y_mean) ** 2 for b in y)
	den = math.sqrt(dx * dy)
	if den == 0.0:
		return float("nan")
	return num / den


def _spearman_corr(x: list[float], y: list[float]) -> float:
	return _pearson_corr(_rankdata(x), _rankdata(y))


def run_stability_experiment(prompt: str, runs: int, samples: int, seed: int | None, progress: bool):
	"""Repeat same prompt end-to-end and collect attribution scores per run.

	Returns:
	  stability_scores: { fragment: { feature: [score_run1, score_run2, ...] } }
	  fragments_per_run: [ [frag1, frag2, ...], ...]
	"""
	stability_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
	fragments_per_run: list[list[str]] = []

	for run_idx in range(runs):
		if seed is not None:
			# Make runs reproducible but different.
			random.seed(seed + run_idx)

		fragments = extract_prompt_fragments(prompt)
		fragments_per_run.append(fragments)

		attributions = shap_sampling_attribution(
			fragments,
			generate_architecture,
			samples=samples,
			progress=progress,
		)

		for frag, feats in attributions.items():
			for feature, score in feats.items():
				stability_scores[frag][feature].append(float(score))

	# Convert defaultdicts to plain dicts.
	stability_scores = {
		frag: {feature: scores for feature, scores in feat_map.items()}
		for frag, feat_map in stability_scores.items()
	}
	return stability_scores, fragments_per_run


def compute_stability_metrics(stability_scores: dict[str, dict[str, list[float]]], runs: int) -> dict:
	"""Compute aggregate stability metrics from stability_scores."""
	std_abs_values: list[float] = []
	std_values: list[float] = []
	mean_abs_values: list[float] = []

	# For rank stability.
	importance_per_run: list[dict[str, float]] = [defaultdict(float) for _ in range(runs)]
	coverage_per_run: list[set[str]] = [set() for _ in range(runs)]

	for frag, feat_map in stability_scores.items():
		for feature, scores in feat_map.items():
			if not scores:
				continue
			abs_scores = [abs(s) for s in scores]

			# Note: lists can be shorter than runs if extraction varies; use what we have.
			if len(scores) >= 2:
				std_values.append(statistics.pstdev(scores))
				std_abs_values.append(statistics.pstdev(abs_scores))
			else:
				std_values.append(0.0)
				std_abs_values.append(0.0)
			mean_abs_values.append(sum(abs_scores) / len(abs_scores))

			for run_idx, score in enumerate(scores[:runs]):
				importance_per_run[run_idx][frag] += abs(score)
				coverage_per_run[run_idx].add(frag)

	# Pairwise Spearman correlation of fragment-importance rankings.
	pairwise_corrs: list[float] = []
	pairwise_overlap_sizes: list[int] = []
	for i in range(runs):
		for j in range(i + 1, runs):
			common = sorted(coverage_per_run[i].intersection(coverage_per_run[j]))
			pairwise_overlap_sizes.append(len(common))
			if len(common) < 2:
				continue
			x = [importance_per_run[i].get(f, 0.0) for f in common]
			y = [importance_per_run[j].get(f, 0.0) for f in common]
			corr = _spearman_corr(x, y)
			if not math.isnan(corr):
				pairwise_corrs.append(corr)

	metrics = {
		"mean_std_abs": float(sum(std_abs_values) / len(std_abs_values)) if std_abs_values else 0.0,
		"mean_std": float(sum(std_values) / len(std_values)) if std_values else 0.0,
		"mean_mean_abs": float(sum(mean_abs_values) / len(mean_abs_values)) if mean_abs_values else 0.0,
		"pairwise_spearman_importance": float(sum(pairwise_corrs) / len(pairwise_corrs)) if pairwise_corrs else None,
		"pairwise_overlap_min": min(pairwise_overlap_sizes) if pairwise_overlap_sizes else 0,
		"pairwise_overlap_mean": float(sum(pairwise_overlap_sizes) / len(pairwise_overlap_sizes)) if pairwise_overlap_sizes else 0.0,
	}
	return metrics


def plot_experiment3(results: list[dict], out_path: str):
	"""Save a simple 2-panel plot: time vs samples, stability vs samples."""
	try:
		import matplotlib.pyplot as plt  # type: ignore
	except Exception:
		print("WARN: matplotlib not available; skipping plot.")
		return

	samples = [r["samples"] for r in results]
	time_mean = [r["time_seconds_mean_per_run"] for r in results]
	mean_std_abs = [r["stability"]["mean_std_abs"] for r in results]
	spearman = [r["stability"]["pairwise_spearman_importance"] for r in results]

	fig, axes = plt.subplots(1, 2, figsize=(12, 4))

	ax = axes[0]
	ax.plot(samples, time_mean, marker="o")
	ax.set_xlabel("samples")
	ax.set_ylabel("Mean time per run (s)")
	ax.set_title("Cost vs samples")
	ax.grid(True, alpha=0.3)

	ax = axes[1]
	ax.plot(samples, mean_std_abs, marker="o", label="mean std(|attr|)")
	# Spearman can be None
	if any(v is not None for v in spearman):
		spearman_y = [float("nan") if v is None else float(v) for v in spearman]
		ax.plot(samples, spearman_y, marker="o", label="avg Spearman(importance)")
	ax.set_xlabel("samples")
	ax.set_ylabel("Stability")
	ax.set_title("Stability vs samples")
	ax.grid(True, alpha=0.3)
	ax.legend()

	fig.tight_layout()
	fig.savefig(out_path)


def main():
	parser = argparse.ArgumentParser(description="Experiment 3: cost vs quality by varying SHAP-inspired samples")
	parser.add_argument("--runs", type=int, default=5, help="Repetitions per samples value")
	parser.add_argument(
		"--samples-list",
		type=str,
		default="1,2,5,10",
		help="Comma-separated samples values to test (e.g. 1,2,5,10,20)",
	)
	parser.add_argument("--seed", type=int, default=0, help="Base RNG seed (set -1 for no seed)")
	parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
	parser.add_argument("--out-json", type=str, default="ex3_results.json", help="Output JSON path")
	parser.add_argument("--out-plot", type=str, default="ex3_cost_vs_quality.png", help="Output plot path")
	parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt text")
	args = parser.parse_args()

	runs = max(1, args.runs)
	seed = None if args.seed == -1 else args.seed
	progress = not args.no_progress

	# Parse samples list.
	samples_list: list[int] = []
	for part in args.samples_list.split(","):
		part = part.strip()
		if not part:
			continue
		try:
			samples_list.append(max(1, int(part)))
		except ValueError:
			raise SystemExit(f"Invalid --samples-list entry: {part!r}")
	# De-dup and sort.
	samples_list = sorted(set(samples_list))
	if not samples_list:
		raise SystemExit("--samples-list produced an empty list")

	results: list[dict] = []

	print("\n=== Experiment 3: cost vs quality (vary samples) ===")
	print(f"runs={runs} | seed={'none' if seed is None else seed} | samples_list={samples_list}")

	for samples in samples_list:
		print(f"\n--- samples={samples} ---")
		t0 = time.perf_counter()
		try:
			stability_scores, fragments_per_run = run_stability_experiment(
				prompt=args.prompt,
				runs=runs,
				samples=samples,
				seed=seed,
				progress=progress,
			)
		except requests.exceptions.RequestException as e:
			print("ERROR: Could not reach Ollama at http://localhost:11434.")
			print("Start Ollama and ensure the model is available, then rerun.")
			print(f"Details: {e}")
			return 2
		t1 = time.perf_counter()

		unique_fragment_sets = {tuple(frags) for frags in fragments_per_run}
		fragment_extraction_unique_sets = len(unique_fragment_sets)
		fragment_extraction_stable = fragment_extraction_unique_sets == 1

		stability = compute_stability_metrics(stability_scores, runs=runs)

		elapsed = t1 - t0
		result = {
			"samples": samples,
			"runs": runs,
			"time_seconds_total": float(elapsed),
			"time_seconds_mean_per_run": float(elapsed / runs),
			"fragment_extraction_stable": bool(fragment_extraction_stable),
			"fragment_extraction_unique_sets": int(fragment_extraction_unique_sets),
			"n_fragments_run0": int(len(fragments_per_run[0])) if fragments_per_run else 0,
			"stability": stability,
		}
		results.append(result)

		print(
			"time_total={:.2f}s | time/run={:.2f}s | frag_sets={} | mean_std_abs={:.4f} | spearman={}"
			.format(
				result["time_seconds_total"],
				result["time_seconds_mean_per_run"],
				result["fragment_extraction_unique_sets"],
				result["stability"]["mean_std_abs"],
				("{:.3f}".format(stability["pairwise_spearman_importance"]) if stability["pairwise_spearman_importance"] is not None else "n/a"),
			)
		)

	# Save JSON
	payload = {
		"prompt": args.prompt,
		"seed": seed,
		"results": results,
	}
	with open(args.out_json, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=False)
	print(f"\nSaved: {args.out_json}")

	# Save plot
	plot_experiment3(results, args.out_plot)
	print(f"Saved: {args.out_plot}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
