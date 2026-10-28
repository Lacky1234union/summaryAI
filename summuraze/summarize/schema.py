"""JSON-схема ответа для structured outputs и пустой документ по ней.

Строгий режим (`strict: true`) у OpenAI-совместимых провайдеров требует, чтобы у каждого объекта
были `additionalProperties: false` и все поля в `required`. Pydantic так не генерирует — дописываем.

Refs #17
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    _strictify(schema)
    return schema


def empty_document(schema: dict[str, Any]) -> Any:
    """Минимальный валидный экземпляр схемы: пустые строки и списки. Нужен заглушке `echo`."""
    defs = schema.get("$defs", {})

    def build(node: dict[str, Any]) -> Any:
        if "$ref" in node:
            return build(defs[node["$ref"].rsplit("/", 1)[-1]])
        kind = node.get("type")
        if kind == "object":
            return {name: build(sub) for name, sub in node.get("properties", {}).items()}
        if kind == "array":
            return []
        if kind in ("integer", "number"):
            return 0
        if kind == "boolean":
            return False
        return ""

    return build(schema)


def _strictify(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}))
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for value in node:
            _strictify(value)
