"""Structured architecture generation and outcome scoring."""

import re

from lcnc import call_ollama, parse_first_json

FRAGMENTS = [
    "System ma obsługiwać 500 równoczesnych użytkowników.",
    "Dane pacjentów muszą być szyfrowane.",
    "Potwierdzenie wizyty jest wysyłane e-mailem.",
]


def generate(prompt: str):
    instruction = f"""Design a component architecture supported by the prompt.
Return only JSON with this shape:
{{"components":[{{"id":"C1","name":"...","responsibilities":["..."]}}],
"connections":[{{"from":"C1","to":"C2","purpose":"..."}}]}}.
Name components and explain relevant responsibilities and connections.
Do not add unsupported functionality or claims.

PROMPT:
{prompt}
"""
    try:
        result = parse_first_json(call_ollama(instruction, response_format="json"))
        if isinstance(result, dict) and isinstance(result.get("components"), list) and isinstance(result.get("connections"), list):
            return result
        return None
    except (ValueError, TypeError):
        return None


def evaluate(result: dict) -> dict:
    if result is None:
        raise ValueError("Architecture generation returned None")
    component_text = " ".join(
        str(component.get("name", "")) + " " + " ".join(component.get("responsibilities", []))
        for component in result.get("components", [])
    ).lower()
    connection_text = " ".join(str(edge.get("purpose", "")) for edge in result.get("connections", [])).lower()
    full_text = component_text + " " + connection_text
    return {
        "scalability_component": int(bool(re.search(r"500|scale|load balanc|horizontal|concurr|r[oó]wnoczes", full_text))),
        "patient_data_encryption": int(bool(re.search(r"encrypt|szyfr|cryptograph|cipher", full_text))),
        "appointment_email_path": int(bool(re.search(r"e-?mail|email", full_text)) and bool(re.search(r"appointment|wizyt|booking|rezerw|notification|powiadom", full_text))),
    }
