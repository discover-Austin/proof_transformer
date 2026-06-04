from __future__ import annotations

from .models import ValidatorResult


def summarize_validator_results(results: list[ValidatorResult]) -> dict:
    passed = [result for result in results if result.passed]
    failed = [result for result in results if not result.passed]
    average = sum(result.score for result in results) / len(results) if results else 0.0
    return {
        "validator_count": len(results),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "average_score": average,
        "results": [result.as_record() for result in results],
    }
