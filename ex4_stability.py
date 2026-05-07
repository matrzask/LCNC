import argparse
import json
import random
from collections import defaultdict

import requests

from lcnc import extract_prompt_fragments, generate_architecture, shap_sampling_attribution
from visualize import plot_stability


DEFAULT_PROMPT = """
The system should allow users to register and log in.
It must also provide a dashboard for managing their profiles and settings.
Additionally, the application should support offline access and send notifications for important updates.
""".strip()


def run_stability_experiment(
	prompt: str,
	runs: int,
	samples: int,
	seed: int | None,
	progress: bool,
):
	"""Repeat the same prompt end-to-end and measure attribution stability.

	Returns stability_scores:
	  { fragment: { feature: [score_run1, score_run2, ...] } }
	"""

	stability_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
	fragments_per_run: list[list[str]] = []

	for run_idx in range(runs):
		if seed is not None:
			# Make runs reproducible but still different from each other.
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

	# Convert defaultdicts to plain dicts for JSON serialization.
	stability_scores = {
		frag: {feature: scores for feature, scores in feat_map.items()}
		for frag, feat_map in stability_scores.items()
	}

	return stability_scores, fragments_per_run


def _fragment_summary(stability_scores: dict[str, dict[str, list[float]]]):
	import math

	summary = []
	for frag, feat_map in stability_scores.items():
		flat = [s for scores in feat_map.values() for s in scores]
		flat_abs = [abs(x) for x in flat]
		if not flat:
			summary.append((frag, 0.0, 0.0, 0))
			continue
		mean_abs = sum(flat_abs) / len(flat_abs)
		var_abs = sum((x - mean_abs) ** 2 for x in flat_abs) / len(flat_abs)
		std_abs = math.sqrt(var_abs)
		summary.append((frag, mean_abs, std_abs, len(flat_abs)))

	# Most unstable first by std_abs, then by mean_abs
	summary.sort(key=lambda t: (t[2], t[1]), reverse=True)
	return summary


def main():
	parser = argparse.ArgumentParser(description="Stability experiment: repeat the same prompt multiple times")
	parser.add_argument("--runs", type=int, default=10, help="Number of repetitions")
	parser.add_argument("--samples", type=int, default=10, help="Subset samples per fragment (SHAP sampling)")
	parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for reproducibility (set -1 for no seed)")
	parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
	parser.add_argument("--out", type=str, default="stability.png", help="Output plot path")
	parser.add_argument(
		"--prompt",
		type=str,
		default=DEFAULT_PROMPT,
		help="Prompt text (default: built-in demo prompt)",
	)
	args = parser.parse_args()

	runs = max(1, args.runs)
	samples = max(1, args.samples)
	seed = None if args.seed == -1 else args.seed
	progress = not args.no_progress

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

	# Console output
	print("\n=== Stability Experiment (repeat same prompt) ===")
	print(f"runs={runs} | samples={samples} | seed={'none' if seed is None else seed}")

	# Basic fragment consistency check across runs
	unique_fragment_sets = {tuple(frags) for frags in fragments_per_run}
	if len(unique_fragment_sets) == 1:
		print(f"fragment extraction: stable (n_fragments={len(fragments_per_run[0])})")
	else:
		print(f"fragment extraction: VARIES across runs (unique_sets={len(unique_fragment_sets)})")

	summary = _fragment_summary(stability_scores)
	if not summary:
		print("No stability data collected.")
		return 1

	print("\nTop fragments by instability (std of |attribution| across all features):")
	for frag, mean_abs, std_abs, n in summary[: min(10, len(summary))]:
		print(f"- std={std_abs:.4f} | mean_abs={mean_abs:.4f} | n={n} | {frag}")

	# Save JSON
	with open("stability_scores.json", "w", encoding="utf-8") as f:
		json.dump(stability_scores, f, indent=2, ensure_ascii=False)

	# Save plot
	fig = plot_stability(stability_scores)
	fig.savefig(args.out)
	print(f"\nSaved: stability scores -> stability_scores.json")
	print(f"Saved: stability plot   -> {args.out}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

