from __future__ import annotations

import sqlite3

from .ledger import LearningLedger


class LedgerVerificationError(RuntimeError):
    pass


def verify_chain_or_raise(conn: sqlite3.Connection) -> dict:
    result = LearningLedger(conn).verify_chain()
    if not result["ok"]:
        raise LedgerVerificationError("; ".join(result["errors"]))
    return result
