"""Requirement generation and deterministic outcome scoring."""

import re

from lcnc import call_ollama, parse_first_json

FRAGMENTS = [
    "Pacjent może umówić wizytę.",
    "Wizytę można odwołać najpóźniej 24 godziny przed terminem.",
    "Po rezerwacji pacjent dostaje e-mail.",
]


def generate(prompt: str):
    instruction = f"""Convert the prompt into atomic, testable software requirements.
Return only JSON: {{"requirements":[{{"id":"R1","statement":"..."}}]}}.
Preserve all rules, actors, and numeric boundaries. Do not add requirements.

PROMPT:
{prompt}
"""
    try:
        result = parse_first_json(call_ollama(instruction, response_format="json"))
        return result if isinstance(result, dict) and isinstance(result.get("requirements"), list) else None
    except (ValueError, TypeError):
        return None


def evaluate(result: dict) -> dict:
    if result is None:
        raise ValueError("Requirement generation returned None")
    text = " ".join(str(item.get("statement", "")) for item in result.get("requirements", []))
    text = text.lower()
    booking = bool(re.search(r"um[oó]wi|rezerw|book|appointment", text))
    cancellation = bool(re.search(r"odwo[lł]|anul|cancel", text)) and bool(re.search(r"24\s*(godz|hour|h\b)", text))
    email = bool(re.search(r"e-?mail|email", text)) and bool(re.search(r"rezerw|potwier|confirmation|booking", text))
    return {
        "appointment_booking_requirement": int(booking),
        "24_hour_cancellation_rule": int(cancellation),
        "booking_email_requirement": int(email),
    }
