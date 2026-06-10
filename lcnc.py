from collections import defaultdict
import random

import requests
import json

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def call_ollama(prompt, temperature=0.0, response_format=None) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    if response_format is not None:
        payload["format"] = response_format

    response = requests.post(OLLAMA_URL, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        # Backward compatibility: some Ollama versions/models may reject the `format` field.
        if response_format is not None and response.status_code == 400:
            payload.pop("format", None)
            response = requests.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
        else:
            raise
    
    return response.json()["response"].strip()


def parse_first_json(text: str):
    """Extract and parse the first JSON object/array from LLM output.

    Handles common cases:
    - leading/trailing commentary
    - fenced code blocks like ```json ... ```
    - extra content after the JSON
    """

    if text is None:
        raise ValueError("No text to parse")

    cleaned = text.strip()
    decoder = json.JSONDecoder()

    # Prefer content inside the first fenced code block, if present.
    if "```" in cleaned:
        parts = cleaned.split("```")
        fenced_candidates = []
        for i in range(1, len(parts), 2):
            fenced_candidates.append(parts[i])

        # Choose the first fenced block that looks like it contains JSON.
        for candidate in fenced_candidates:
            c = candidate.strip()
            if not c:
                continue
            first_line = c.splitlines()[0].strip().lower()
            if first_line in {"json", "javascript", "js"}:
                c = "\n".join(c.splitlines()[1:]).strip()
            if "{" in c or "[" in c:
                cleaned = c
                break

    # Find first JSON start char and decode from there.
    obj_start = None
    for ch in ("{", "["):
        pos = cleaned.find(ch)
        if pos != -1:
            obj_start = pos if obj_start is None else min(obj_start, pos)

    if obj_start is None:
        raise ValueError("No JSON object/array start found")

    parsed, _end = decoder.raw_decode(cleaned[obj_start:])
    return parsed

def extract_prompt_fragments(user_prompt: str) -> list:
    """
    Dzieli złożony prompt użytkownika na mniejsze fragmenty,
    z których każdy odpowiada jednej funkcjonalności lub wymaganiu.
    """
    decomposition_prompt = f"""
    You are a system that extracts requirement fragments from a user prompt.

    STRICT RULES (CRITICAL):
    - DO NOT paraphrase.
    - DO NOT summarize.
    - DO NOT translate.
    - DO NOT reorder words inside fragments.
    - DO NOT merge multiple requirements into one.
    - Each fragment must correspond to ONE distinct requirement or feature.
    - Keep fragments as close as possible to the original wording.
    - If a sentence contains multiple requirements, split it into smaller parts.

    OUTPUT FORMAT:
    - Return ONLY fragments
    - One fragment per line
    - NO bullets
    - NO numbering
    - NO explanations

    USER PROMPT:
    \"\"\"{user_prompt}\"\"\"
    """

    raw_output = call_ollama(decomposition_prompt, temperature=0.0)

    # czyszczenie
    fragments = []
    for line in raw_output.split("\n"):
        line = line.strip()

        if line:
            fragments.append(line)

    return fragments

def generate_architecture(user_prompt: str) -> dict:
    """
    Generuje strukturę architektury low-code/no-code na podstawie promptu użytkownika,
    zgodnie z ściśle określonymi regułami i formatem JSON.
    """

    architecture_prompt = """
    You are a system that generates a STRUCTURED low-code/no-code application architecture from a user prompt.

    Your output will be used for automated analysis, so it must be STRICT, COMPLETE, and MACHINE-READABLE.

    ========================================
    CRITICAL RULES (DO NOT VIOLATE)
    ========================================
    - Output MUST be valid JSON.
    - DO NOT include explanations.
    - DO NOT include comments.
    - DO NOT include any text outside JSON.
    - DO NOT infer features not explicitly supported by the prompt.
    - DO NOT add "best practices" or optional features.
    - ONLY include elements that are clearly implied by the prompt.
    - If something is not mentioned, DO NOT include it.

    ========================================
    OUTPUT FORMAT (STRICT JSON)
    ========================================

    {{
    "entities": [
        {{
        "name": "EntityName",
        "attributes": ["attribute1", "attribute2"]
        }}
    ],
    "relationships": [
        {{
        "from": "EntityA",
        "to": "EntityB",
        "type": "relationship_type"
        }}
    ],
    "operations": [
        {{
        "type": "operation_type",
        "entity": "EntityName"
        }}
    ],
    "features": {{
        "has_offline_storage": 0,
        "has_reminders": 0,
        "has_notifications": 0,
        "has_authentication": 0,
        "has_sharing": 0,
        "has_search": 0
    }},
    "architecture": {{
        "uses_local_storage": 0,
        "uses_database": 0,
        "uses_api": 0
    }}
    }}

    ========================================
    FEATURE DEFINITIONS (IMPORTANT)
    ========================================
    Set values to 1 ONLY if clearly supported by the prompt.

    - has_offline_storage → app works without internet / offline mode
    - has_reminders → user gets reminders (time-based)
    - has_notifications → push/system notifications
    - has_authentication → login/user accounts
    - has_sharing → sharing data with others
    - has_search → ability to search data

    ARCHITECTURE:
    - uses_local_storage → data stored locally on device
    - uses_database → persistent structured storage
    - uses_api → external or backend API communication

    ========================================
    OPERATION RULES
    ========================================
    Include operations ONLY if implied.
    Operation types should be standardized as:
    - create
    - read
    - update
    - delete
    - notify (only if reminders/notifications exist)
    Also include the associated entity for each operation.

    ========================================
    ENTITY RULES
    ========================================
    - Only include entities clearly present in the prompt.
    - Use simple names (User, Task, Note, etc.)
    - Do not invent entities.

    ========================================
    USER PROMPT
    ========================================
    \"\"\"{user_prompt}\"\"\"
    """.format(user_prompt=user_prompt)

    raw_output = call_ollama(architecture_prompt, temperature=0.0, response_format="json")

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        try:
            return parse_first_json(raw_output)
        except Exception:
            return None

def extract_feature_vector(arch: dict) -> dict:
    """
    Zamienia architekturę JSON na płaski feature vector.
    """

    features = arch.get("features", {})
    architecture = arch.get("architecture", {})
    entities = arch.get("entities", [])
    relationships = arch.get("relationships", [])
    operations = arch.get("operations", [])

    feature_vector = {
        # === funkcjonalne ===
        "has_offline_storage": int(features.get("has_offline_storage", 0)),
        "has_reminders": int(features.get("has_reminders", 0)),
        "has_notifications": int(features.get("has_notifications", 0)),
        "has_authentication": int(features.get("has_authentication", 0)),
        "has_sharing": int(features.get("has_sharing", 0)),
        "has_search": int(features.get("has_search", 0)),

        # === strukturalne ===
        "num_entities": len(entities),
        "num_relationships": len(relationships),
        "has_user_entity": int(any(e["name"].lower() == "user" for e in entities)),
        "has_task_entity": int(any(e["name"].lower() == "task" for e in entities)),

        # === operacyjne ===
        "has_create": int(any(op["type"] == "create" for op in operations)),
        "has_read": int(any(op["type"] == "read" for op in operations)),
        "has_update": int(any(op["type"] == "update" for op in operations)),
        "has_delete": int(any(op["type"] == "delete" for op in operations)),
        "has_notify_operation": int(any(op["type"] == "notify" for op in operations)),

        # === architektoniczne ===
        "uses_local_storage": int(architecture.get("uses_local_storage", 0)),
        "uses_database": int(architecture.get("uses_database", 0)),
        "uses_api": int(architecture.get("uses_api", 0)),
    }

    return feature_vector

def feature_diff(f1: dict, f2: dict):
    """
    Różnica feature’ów: f1 - f2
    Interpretacja:
    1 → feature obecny w f1, nieobecny w f2 (dodany przez frag)
    -1 → feature obecny w f2, nieobecny w f1 (usunięty przez frag)
    0 → feature obecny w obu lub w żadnym (niezmieniony przez frag)
    """
    diff = {}
    for k in f1:
        diff[k] = f1[k] - f2.get(k, 0)
    return diff

def shap_feature_attribution_oneout(prompt: str, fragments: list, generate_arch_fn: callable) -> dict:
    """
    SHAP-like attribution: leave-one-out
    """

    # baseline
    full_arch = generate_arch_fn(prompt)
    full_features = extract_feature_vector(full_arch)

    attributions = {}

    for frag in fragments:
        reduced_fragments = [f for f in fragments if f != frag]
        reduced_prompt = " ".join(reduced_fragments)

        reduced_arch = generate_arch_fn(reduced_prompt)
        reduced_features = extract_feature_vector(reduced_arch)

        diff = feature_diff(full_features, reduced_features)

        attributions[frag] = diff

    return attributions

def shap_sampling_attribution(fragments, generate_arch_fn, samples=10, progress=True):
    """
    SHAP-inspired attribution (sampling subsets)
    """

    contributions = {frag: defaultdict(float) for frag in fragments}

    frag_iter = fragments
    if progress and tqdm is not None:
        frag_iter = tqdm(fragments, desc="Attributing fragments", unit="frag")

    for frag in frag_iter:
        other_frags = [f for f in fragments if f != frag]

        sample_iter = range(samples)
        if progress and tqdm is not None:
            sample_iter = tqdm(sample_iter, desc="Sampling subsets", unit="sample", leave=False)

        for _ in sample_iter:
            subset_size = random.randint(0, len(other_frags))
            subset = random.sample(other_frags, subset_size)

            prompt_without = " ".join(subset)
            prompt_with = " ".join(subset + [frag])

            arch_without = generate_arch_fn(prompt_without)
            arch_with = generate_arch_fn(prompt_with)
            
            if arch_without is None or arch_with is None:
                continue  # pomiń próbki z błędami parsowania

            f_without = extract_feature_vector(arch_without)
            f_with = extract_feature_vector(arch_with)

            diff = feature_diff(f_with, f_without)

            for k, v in diff.items():
                contributions[frag][k] += v

        # uśrednianie
        for k in contributions[frag]:
            contributions[frag][k] /= samples

    return contributions


if __name__ == "__main__":
    user_prompt = """
    The system should allow users to register and log in.
    Each user will be able to create and manage their own tasks.
    It must also provide a dashboard for managing their profiles and settings.
    Additionally, the application should support offline access and send notifications for important updates.
    """

    fragments = extract_prompt_fragments(user_prompt)
    # arch = generate_architecture(user_prompt)
    # print("Generated architecture:")
    # print(json.dumps(arch, indent=2))
    attributions = shap_sampling_attribution(fragments, generate_architecture, samples=5)

    print(json.dumps(attributions, indent=2))

    with open("attributions.json", "w") as f:
        json.dump(attributions, f, indent=2)