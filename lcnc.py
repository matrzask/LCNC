import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

def call_ollama(prompt, temperature=0.0):
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

def extract_prompt_fragments(user_prompt: str):
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

def generate_architecture(user_prompt: str):
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


if __name__ == "__main__":
    user_prompt = "The system should allow users to register and log in. It must also provide a dashboard for managing their profiles and settings."
    # fragments = extract_prompt_fragments_ollama(user_prompt)
    # for i, fragment in enumerate(fragments, 1):
    #     print(f"Fragment {i}: {fragment}")
    out = generate_architecture(user_prompt)
    print(out)