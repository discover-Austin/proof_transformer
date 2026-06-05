from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proof_transformer.timeutils import parse_utc, utc_now


class TimeutilsTests(unittest.TestCase):
    def test_utc_now_is_strictly_monotonic_for_fast_local_loops(self) -> None:
        values = [utc_now() for _ in range(1000)]

        self.assertEqual(values, sorted(values))
        self.assertEqual(len(values), len(set(values)))
        for earlier, later in zip(values, values[1:]):
            self.assertLess(parse_utc(earlier), parse_utc(later))

    def test_utc_now_is_unique_for_concurrent_local_loops(self) -> None:
        with ThreadPoolExecutor(max_workers=16) as executor:
            values = list(executor.map(lambda _: utc_now(), range(1000)))

        parsed = sorted(parse_utc(value) for value in values)
        self.assertEqual(len(values), len(set(values)))
        for earlier, later in zip(parsed, parsed[1:]):
            self.assertLess(earlier, later)


if __name__ == "__main__":
    unittest.main()
