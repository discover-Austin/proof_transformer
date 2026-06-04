from __future__ import annotations

import time

from .orchestrator import ProofTransformerOrchestrator


class QueueWorker:
    def __init__(self, orchestrator: ProofTransformerOrchestrator, poll_interval: float = 2.0) -> None:
        self.orchestrator = orchestrator
        self.poll_interval = max(float(poll_interval), 0.0)

    def run_once(self, limit: int | None = None) -> dict:
        return self.orchestrator.run_queue(limit=limit)

    def serve(self, max_cycles: int | None = None, limit_per_cycle: int | None = None) -> dict:
        cycles = 0
        total_ran = 0
        history: list[dict] = []
        while max_cycles is None or cycles < max_cycles:
            cycle_result = self.run_once(limit=limit_per_cycle)
            total_ran += int(cycle_result.get("ran", 0))
            history.append(cycle_result)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            time.sleep(self.poll_interval)
        return {
            "cycles": cycles,
            "total_ran": total_ran,
            "poll_interval": self.poll_interval,
            "history": history,
        }
