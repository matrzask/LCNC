import requests

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

def extract_prompt_fragments_ollama(user_prompt: str):
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

if __name__ == "__main__":
    user_prompt = "The system should allow users to register and log in. It must also provide a dashboard for managing their profiles and settings."
    fragments = extract_prompt_fragments_ollama(user_prompt)
    for i, fragment in enumerate(fragments, 1):
        print(f"Fragment {i}: {fragment}")