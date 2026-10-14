from pydantic import BaseModel


def strict_json_schema(model: type[BaseModel]) -> dict:
    """
    Pydantic's schema, adjusted for strict structured output.

    Strict mode wants every property listed as required and no extra keys.
    An optional field stays optional by allowing null, not by being absent.
    """
    schema = model.model_json_schema()

    def fix(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"].keys())
                node["additionalProperties"] = False
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema
