"""Run one three-fragment domain pilot and print its Shapley attribution."""

import argparse
import json

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", choices=DOMAINS)
    parser.add_argument("--samples", type=int, help="Coalitions per fragment; omit for exact Shapley")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", help="Override the Ollama model name")
    args = parser.parse_args()

    if args.model:
        lcnc.MODEL = args.model
    domain = DOMAINS[args.domain]
    fragments = make_fragments(domain.FRAGMENTS)
    scores = shap_attribution(
        fragments,
        domain.generate,
        domain.evaluate,
        samples=args.samples,
        seed=args.seed,
    )
    print(json.dumps({
        "domain": args.domain,
        "model": lcnc.MODEL,
        "samples_per_fragment": args.samples,
        "fragments": [
            {"id": fragment.id, "text": str(fragment), "attribution": scores[fragment.id]}
            for fragment in fragments
        ],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
