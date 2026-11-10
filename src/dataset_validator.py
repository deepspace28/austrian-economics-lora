"""Validate generated instruction-tuning JSONL before training."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


SUPPORTED_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True)
class DatasetIssue:
    line_number: int
    message: str


@dataclass(frozen=True)
class ValidationResult:
    records: int
    valid: int
    issues: tuple[DatasetIssue, ...]

    @property
    def ok(self) -> bool:
        return self.records > 0 and self.valid == self.records


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_sample(sample: Any) -> list[str]:
    """Return schema errors for one record, or an empty list when it is valid."""
    if not isinstance(sample, dict):
        return ["record must be a JSON object"]

    if "messages" in sample:
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            return ["'messages' must contain at least two messages"]

        errors = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"messages[{index}] must be an object")
                continue
            if message.get("role") not in SUPPORTED_ROLES:
                errors.append(
                    f"messages[{index}].role must be one of "
                    f"{sorted(SUPPORTED_ROLES)}"
                )
            if not _has_text(message.get("content")):
                errors.append(f"messages[{index}].content must be non-empty text")

        last_message = messages[-1]
        if not isinstance(last_message, dict) or last_message.get("role") != "assistant":
            errors.append("last message must have role 'assistant'")
        return errors

    if "instruction" in sample or "response" in sample:
        errors = []
        if not _has_text(sample.get("instruction")):
            errors.append("'instruction' must be non-empty text")
        if not _has_text(sample.get("response")):
            errors.append("'response' must be non-empty text")
        return errors

    return ["expected 'messages' or 'instruction'/'response' schema"]


def validate_dataset(path: Path, max_issues: int = 20) -> ValidationResult:
    """Validate non-empty JSONL records and retain the first few useful errors."""
    if isinstance(max_issues, bool) or not isinstance(max_issues, int):
        raise TypeError('max_issues must be an integer')
    if max_issues < 0:
        raise ValueError("max_issues must be non-negative")

    records = 0
    valid = 0
    issues = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            records += 1
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                if len(issues) < max_issues:
                    issues.append(
                        DatasetIssue(line_number, f"invalid JSON: {error.msg}")
                    )
                continue

            errors = validate_sample(sample)
            if errors:
                if len(issues) < max_issues:
                    issues.append(DatasetIssue(line_number, "; ".join(errors)))
                continue
            valid += 1

    if records == 0 and max_issues > 0:
        issues.append(DatasetIssue(0, "dataset contains no records"))

    return ValidationResult(records, valid, tuple(issues))
