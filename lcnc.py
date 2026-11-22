"""Shared utilities for prompt-fragment attribution across output domains."""

import itertools
import json
import math
import random

import requests

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:4b"
MAX_PREDICT_TOKENS = 2048


class PromptFragment(str):
    """Prompt text with a stable identity within one prompt (P1, P2, ...)."""

    def __new__(cls, fragment_id: str, text: str):
        instance = super().__new__(cls, text)
        instance.id = fragment_id
        return instance

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, PromptFragment) and self.id == other.id


def make_fragments(texts):
    """Create identified fragments from manually specified prompt sentences."""
    return [PromptFragment(f"P{i}", text) for i, text in enumerate(texts, 1)]


def call_ollama(prompt, temperature=0.0, response_format=None, think=False) -> str:
    """Call Ollama with reasoning disabled by default."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": MAX_PREDICT_TOKENS,
        },
    }
    if response_format is not None:
        payload["format"] = response_format
    if think is not None:
        payload["think"] = think
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        if response_format is not None and response.status_code == 400:
            payload.pop("format", None)
            response = requests.post(OLLAMA_URL, json=payload, timeout=180)
            response.raise_for_status()
        else:
            raise
    return response.json()["response"].strip()


def parse_first_json(text: str):
    """Parse the first JSON object or array from an LLM response."""
    if text is None:
        raise ValueError("No text to parse")
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for candidate in parts[1::2]:
            candidate = candidate.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate.split("\n", 1)[-1].strip()
            if "{" in candidate or "[" in candidate:
                cleaned = candidate
                break
    starts = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
    if not starts:
        raise ValueError("No JSON object/array start found")
    parsed, _ = json.JSONDecoder().raw_decode(cleaned[min(starts):])
    return parsed


def extract_prompt_fragments(user_prompt: str) -> list[PromptFragment]:
    """Split a prompt into atomic, source-faithful fragments and assign IDs."""
    prompt = f"""Extract atomic requirements from the user prompt.
Preserve the original language and wording. Do not merge separate requirements.
Return only one fragment per line, with no numbering or commentary.

USER PROMPT:
{user_prompt}
"""
    raw = call_ollama(prompt)
    fragments = [line.strip().lstrip("-• ").strip() for line in raw.splitlines()]
    fragments = [line for line in fragments if line]
    return make_fragments(fragments)


def _normalise_fragments(fragments):
    identified = [
        (getattr(fragment, "id", f"P{i}"), str(fragment))
        for i, fragment in enumerate(fragments, 1)
    ]
    ids = [fragment_id for fragment_id, _ in identified]
    if len(set(ids)) != len(ids):
        raise ValueError("Fragment IDs must be unique")
    return identified


def shap_attribution(
    fragments,
    generate_fn,
    evaluate_fn,
    samples: int | None = None,
    progress: bool = True,
    progress_callback=None,
    seed: int | None = None,
) -> dict:
    """Compute exact Shapley values, or a sampled estimate for larger prompts.

    The evaluator must return a numeric mapping from outcome IDs to scores.
    Coalition results are cached within this call. Failed generations are
    skipped and successful comparison weights are renormalized per outcome.
    """
    identified = _normalise_fragments(fragments)
    ids = [fragment_id for fragment_id, _ in identified]
    texts = dict(identified)
    n = len(ids)
    if samples is not None and samples <= 0:
        raise ValueError("samples must be positive or None for exact enumeration")
    rng = random.Random(seed)
    cache = {}

    def coalition_value(coalition):
        key = frozenset(coalition)
        if key not in cache:
            prompt = " ".join(texts[item] for item in ids if item in key)
            try:
                result = generate_fn(prompt)
                value = None if result is None else evaluate_fn(result)
                if value is not None:
                    if not isinstance(value, dict):
                        raise TypeError("evaluate_fn must return a dict or None")
                    if any(not isinstance(v, (int, float)) for v in value.values()):
                        raise TypeError("Outcome scores must be numeric")
                cache[key] = value
            except Exception:
                cache[key] = None
        return cache[key]

    contributions = {fragment_id: {} for fragment_id in ids}
    work_total = n * (2 ** (n - 1) if samples is None and n else (samples or 0))
    completed = 0
    iterator = tqdm(ids, desc="Attributing fragments", unit="fragment") if progress and tqdm else ids

    for fragment_index, fragment_id in enumerate(iterator, 1):
        others = [item for item in ids if item != fragment_id]
        if samples is None:
            subsets = [
                set(coalition)
                for size in range(n)
                for coalition in itertools.combinations(others, size)
            ]
            weighted_subsets = [
                (
                    subset,
                    math.factorial(len(subset))
                    * math.factorial(n - len(subset) - 1)
                    / math.factorial(n),
                )
                for subset in subsets
            ]
        else:
            weighted_subsets = []
            for _ in range(samples):
                size = rng.randint(0, len(others))
                weighted_subsets.append((set(rng.sample(others, size)), 1 / samples))

        totals, successful_weights = {}, {}
        for sample_index, (subset, weight) in enumerate(weighted_subsets, 1):
            before = coalition_value(subset)
            after = coalition_value(subset | {fragment_id})
            if before is not None and after is not None:
                for outcome in before.keys() | after.keys():
                    delta = after.get(outcome, 0) - before.get(outcome, 0)
                    totals[outcome] = totals.get(outcome, 0.0) + delta * weight
                    successful_weights[outcome] = successful_weights.get(outcome, 0.0) + weight
            completed += 1
            if progress_callback:
                progress_callback(
                    completed, work_total, fragment_id, fragment_index, n,
                    sample_index, len(weighted_subsets),
                )
        contributions[fragment_id] = {
            outcome: total / successful_weights[outcome]
            for outcome, total in totals.items()
            if successful_weights[outcome]
        }
    return contributions
