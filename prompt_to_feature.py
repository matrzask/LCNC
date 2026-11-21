"""Legacy LC/NC architecture generation and prompt-to-feature evaluation."""

import json

from lcnc import call_ollama, parse_first_json

FRAGMENTS = [
    "Users can create and manage tasks.",
    "Users can search for tasks.",
    "The application supports offline access.",
]


def generate_architecture(prompt: str):
    """Generate an LC/NC application model using the first-semester schema."""
    schema = {
        "entities": [{"name": "Entity", "attributes": ["attribute"]}],
        "relationships": [{"from": "EntityA", "to": "EntityB", "type": "relationship"}],
        "operations": [{"type": "create|read|update|delete|notify", "entity": "Entity"}],
        "features": {
            "has_offline_storage": 0, "has_reminders": 0,
            "has_notifications": 0, "has_authentication": 0,
            "has_sharing": 0, "has_search": 0,
        },
        "architecture": {"uses_local_storage": 0, "uses_database": 0, "uses_api": 0},
    }
    instruction = f"""Generate a low-code/no-code application model from the prompt.
Return only valid JSON matching this schema, with no unsupported features:
{json.dumps(schema)}

USER PROMPT:
{prompt}
"""
    raw = call_ollama(instruction, response_format="json")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = parse_first_json(raw)
    return result if isinstance(result, dict) else None


def evaluate_features(architecture: dict) -> dict:
    """Flatten the historical LC/NC output into its original feature vector."""
    if architecture is None:
        raise ValueError("Architecture generation returned None")
    features = architecture.get("features", {})
    design = architecture.get("architecture", {})
    entities = architecture.get("entities", [])
    relationships = architecture.get("relationships", [])
    operations = architecture.get("operations", [])
    entity_names = [str(entity.get("name", "")).lower() for entity in entities]
    operation_types = [str(operation.get("type", "")).lower() for operation in operations]
    return {
        **{key: int(features.get(key, 0)) for key in (
            "has_offline_storage", "has_reminders", "has_notifications",
            "has_authentication", "has_sharing", "has_search",
        )},
        "num_entities": len(entities),
        "num_relationships": len(relationships),
        "has_user_entity": int("user" in entity_names),
        "has_task_entity": int("task" in entity_names),
        **{f"has_{operation}": int(operation in operation_types)
           for operation in ("create", "read", "update", "delete")},
        "has_notify_operation": int("notify" in operation_types),
        **{key: int(design.get(key, 0)) for key in (
            "uses_local_storage", "uses_database", "uses_api",
        )},
    }


def feature_diff(first: dict, second: dict) -> dict:
    """Return numeric feature changes (first minus second)."""
    return {key: first.get(key, 0) - second.get(key, 0) for key in first.keys() | second.keys()}


generate = generate_architecture
evaluate = evaluate_features
