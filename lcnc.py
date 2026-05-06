import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def call_ollama(prompt, temperature=0.0) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    
    return response.json()["response"].strip()

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

    # czyszczenie (bardzo ważne)
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
        "name": "operation_name",
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
    Include operations ONLY if implied:
    - create
    - read
    - update
    - delete
    - notify (only if reminders/notifications exist)

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

    raw_output = call_ollama(architecture_prompt, temperature=0.0)

    try:
        parsed_output = json.loads(raw_output)

    except json.JSONDecodeError:
        print("Invalid JSON output:")
        print(raw_output)
        return None

    return parsed_output

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
        "has_create": int(any(op["name"] == "create" for op in operations)),
        "has_read": int(any(op["name"] == "read" for op in operations)),
        "has_update": int(any(op["name"] == "update" for op in operations)),
        "has_delete": int(any(op["name"] == "delete" for op in operations)),
        "has_notify_operation": int(any(op["name"] == "notify" for op in operations)),

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


if __name__ == "__main__":
    user_prompt = "The system should allow users to register and log in. It must also provide a dashboard for managing their profiles and settings."

    fragments = extract_prompt_fragments(user_prompt)
    attributions = shap_feature_attribution_oneout(user_prompt, fragments, generate_architecture)

    print(json.dumps(attributions, indent=2))