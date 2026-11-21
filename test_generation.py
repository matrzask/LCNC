"""Test-case generation and outcome scoring."""

import re

from lcnc import call_ollama, parse_first_json

FRAGMENTS = [
    "Rezerwacja zapisuje wizytę.",
    "Odwołanie jest dozwolone do 24 godzin przed wizytą.",
    "Po rezerwacji pacjent dostaje e-mail.",
]


def generate(prompt: str):
    instruction = f"""Create concise, executable-style test cases from the prompt.
Return only JSON: {{"tests":[{{"name":"...","scenario":"...","expected_result":"..."}}]}}.
Include boundary cases for numeric rules. Do not invent behaviors.

PROMPT:
{prompt}
"""
    try:
        result = parse_first_json(call_ollama(instruction, response_format="json"))
        return result if isinstance(result, dict) and isinstance(result.get("tests"), list) else None
    except (ValueError, TypeError):
        return None


def evaluate(result: dict) -> dict:
    if result is None:
        raise ValueError("Test generation returned None")
    texts = [
        " ".join(str(test.get(key, "")) for key in ("name", "scenario", "expected_result")).lower()
        for test in result.get("tests", [])
    ]
    booking_saved = any(
        re.search(r"rezerw|book", text) and re.search(r"zapis|save|persist|stored", text)
        for text in texts
    )
    accepted_24h = any(
        re.search(r"24\s*(godz|hour|h\b)", text)
        and re.search(r"akcept|dozwol|przyj|accept|allow", text)
        and re.search(r"odwo[lł]|anul|cancel", text)
        for text in texts
    )
    rejected_23h = any(
        re.search(r"23\s*(godz|hour|h\b)", text)
        and re.search(r"odrz|zabron|reject|den|not allow", text)
        and re.search(r"odwo[lł]|anul|cancel", text)
        for text in texts
    )
    email_sent = any(
        re.search(r"e-?mail|email", text)
        and re.search(r"wys[lł]|send|otrzym|receive|deliver", text)
        for text in texts
    )
    return {
        "booking_persists_appointment": int(booking_saved),
        "cancellation_at_24h_accepted": int(accepted_24h),
        "cancellation_at_23h_rejected": int(rejected_23h),
        "booking_sends_email": int(email_sent),
    }
