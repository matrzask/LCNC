"""Run one three-fragment domain pilot and print its Shapley attribution."""

import argparse
import json
import sys

import architecture_design
import prompt_to_feature
import requirements_engineering
import test_generation
import lcnc
from lcnc import make_fragments, shap_attribution

DOMAINS = {
    "prompt-to-feature": prompt_to_feature,
    "requirements": requirements_engineering,
    "architecture": architecture_design,
    "testing": test_generation,
}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=DOMAINS)
    parser.add_argument("--samples", type=int, help="Coalitions per fragment; omit for exact Shapley")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", help="Override the Ollama model name")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum generated tokens per coalition (default: 2048)")
    args = parser.parse_args()

    if args.model:
        lcnc.MODEL = args.model
    lcnc.MAX_PREDICT_TOKENS = args.max_tokens
    domain = DOMAINS[args.domain]
    fragments = make_fragments(domain.FRAGMENTS)

    execution = {
        "generator_calls": 0,
        "generated_outputs": 0,
        "generator_exceptions": 0,
        "evaluation_calls": 0,
        "evaluation_exceptions": 0,
    }

    def tracked_generate(prompt):
        execution["generator_calls"] += 1
        try:
            result = domain.generate(prompt)
        except Exception:
            execution["generator_exceptions"] += 1
            raise
        execution["generated_outputs"] += int(result is not None)
        return result

    def tracked_evaluate(result):
        execution["evaluation_calls"] += 1
        try:
            return domain.evaluate(result)
        except Exception:
            execution["evaluation_exceptions"] += 1
            raise

    scores = shap_attribution(
        fragments,
        tracked_generate,
        tracked_evaluate,
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps({
        "domain": args.domain,
        "model": lcnc.MODEL,
        "max_predict_tokens": args.max_tokens,
        "thinking_enabled": False,
        "samples_per_fragment": args.samples,
        "execution": execution,
        "fragments": [
            {"id": fragment.id, "text": str(fragment), "attribution": scores[fragment.id]}
            for fragment in fragments
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
