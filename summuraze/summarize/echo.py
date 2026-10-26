"""Заглушка LLM: никуда не ходит, возвращает сводку о том, что ей передали.

Нужна этапу 0 и тестам — сквозной прогон без сети и без ключа. Токены оцениваются грубо
(4 символа на токен): цифры видны в `summuraze log`, но ни на что не влияют.
"""

from __future__ import annotations

from summuraze.types import LLMResult, TokenUsage


class EchoLLM:
    name = "echo"
    model = "echo"
    context_window = 1_000_000

    def complete(
        self,
        *,
        system: str,
        prefix: str,
        question: str,
        max_tokens: int,
    ) -> LLMResult:
        lines = [ln for ln in prefix.splitlines() if ln.strip()]
        text = (
            f"Эхо-саммари: транскрипт из {len(lines)} строк, {len(prefix)} символов.\n"
            f"Первая строка: {lines[0] if lines else '—'}"
        )
        return LLMResult(
            text=text,
            model=self.model,
            usage=TokenUsage(
                input=(len(system) + len(prefix) + len(question)) // 4, output=len(text) // 4
            ),
            cost_usd=0.0,
        )
