# tests/test_core.py
import unittest
from pathlib import Path
from dbs_eeg_sync.core import sync_run

class TestCore(unittest.TestCase):
    def test_sync_run_smoke(self):
        eeg = Path("data/eeg_example.set")
        dbs = Path("data/dbs_example.json")
        res = sync_run(sub_id="Sx", block="B1", eeg_file=eeg, dbs_file=dbs)
        self.assertGreaterEqual(res.eeg_sync_idx, 0)
        self.assertGreaterEqual(res.dbs_sync_idx, 0)
        self.assertIsNotNone(res.synchronized_eeg)
        self.assertIsNotNone(res.synchronized_dbs)

if __name__ == "__main__":
    unittest.main()