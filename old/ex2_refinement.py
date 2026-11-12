import argparse
import json
import time
from pprint import pprint

from lcnc import (
    extract_prompt_fragments,
    shap_sampling_attribution,
    generate_architecture,
    extract_feature_vector,
    feature_diff,
)


def read_multiline(prompt_text="Enter prompt (end with empty line):"):
    print(prompt_text)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines).strip()


def run_refinement(initial_prompt: str, samples: int, max_iters: int, target_feature: str | None):
    results = {
        "initial_prompt": initial_prompt,
        "iterations": [],
        "start_time": time.time(),
    }

    print("\nExtracting fragments...")
    fragments = extract_prompt_fragments(initial_prompt)
    print(f"Found {len(fragments)} fragments:\n")
    for i, f in enumerate(fragments, 1):
        print(f"{i}. {f}")

    print("\nComputing SHAP-inspired attributions (sampling)...")
    contributions = shap_sampling_attribution(fragments, generate_architecture, samples=samples, progress=False)

    print("\nAttributions (averaged contribution per feature):")
    pprint(contributions)

    # baseline architecture
    arch = generate_architecture(initial_prompt)
    baseline_vec = extract_feature_vector(arch) if arch is not None else None

    if baseline_vec is None:
        print("Warning: could not parse baseline architecture. Continuing anyway.")
    else:
        print("\nBaseline feature vector:")
        pprint(baseline_vec)

    missing_features = []
    if baseline_vec is not None:
        missing_features = [k for k, v in baseline_vec.items() if isinstance(v, int) and v == 0]

    if target_feature is None:
        print("\nDetected missing features (candidates to target):")
        for k in missing_features:
            print(f"- {k}")
        print("\nYou can optionally supply a target feature to aim for during refinement.")
    else:
        print(f"\nTarget feature: {target_feature}")

    prompt = initial_prompt
    for it in range(1, max_iters + 1):
        print(f"\n--- Iteration {it} ---")
        new_prompt = read_multiline("Paste your refined prompt (end with empty line). Leave empty to stop:")
        if not new_prompt:
            print("Stopping refinement loop.")
            break

        arch_new = generate_architecture(new_prompt)
        if arch_new is None:
            print("Could not parse architecture for the refined prompt. Try again.")
            continue

        vec_new = extract_feature_vector(arch_new)
        diff = feature_diff(vec_new, baseline_vec or {})

        iteration_record = {
            "iteration": it,
            "prompt": new_prompt,
            "feature_vector": vec_new,
            "diff_vs_baseline": diff,
            "timestamp": time.time(),
        }

        results["iterations"].append(iteration_record)

        print("Feature vector for refined prompt:")
        pprint(vec_new)
        print("Difference vs baseline (new - baseline):")
        pprint(diff)

        # check target
        success = False
        if target_feature:
            if vec_new.get(target_feature, 0) and (not baseline_vec or baseline_vec.get(target_feature, 0) == 0):
                print(f"Target feature '{target_feature}' achieved!")
                success = True
            else:
                print(f"Target feature '{target_feature}' not achieved yet.")

        # if user added any previously-missing binary feature, report
        if baseline_vec:
            newly_added = [k for k, v in vec_new.items() if isinstance(v, int) and v == 1 and baseline_vec.get(k, 0) == 0]
            if newly_added:
                print("Newly added features:")
                for k in newly_added:
                    print(f"+ {k}")

        if success:
            print("Refinement successful. Ending loop.")
            break

    results["end_time"] = time.time()
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: prompt refinement loop using SHAP-inspired attributions")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples for SHAP-inspired attribution")
    parser.add_argument("--max-iters", type=int, default=5, help="Maximum refinement iterations")
    parser.add_argument("--target", type=str, default=None, help="Optional target feature to aim for (e.g. has_offline_storage)")
    args = parser.parse_args()

    print("Experiment 2 — Prompt refinement interactive script")
    initial_prompt = read_multiline()
    if not initial_prompt:
        print("No prompt provided. Exiting.")
        return

    results = run_refinement(initial_prompt, samples=args.samples, max_iters=args.max_iters, target_feature=args.target)

    out_path = "ex2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
