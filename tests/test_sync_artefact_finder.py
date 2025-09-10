import unittest
import mne
import pandas as pd
import numpy as np
import numbers

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source.power_calculater import compute_samplewise_eeg_power
from source.sync_artefact_finder import detect_dbs_sync_artifact, detect_eeg_sync_artifact, confirm_sync_selection


class TestArtefactFinderFunctions(unittest.TestCase):

    def test_detect_eeg_sync_artifact(self):
        rng = np.random.default_rng(42)
        raw = mne.io.RawArray(
            rng.standard_normal((9, 130000)),
            mne.create_info(
                ch_names=['POz', 'CPz', 'Pz', 'Oz', 'Fz', 'Cz', 'T7', 'T8', 'O1'],
                sfreq=1000,
                ch_types=['eeg'] * 9,
            )
        )
        # Call the updated function that returns exactly 5 values
        sync_channel, sync_idx, sync_time, sync_result, sync_power = detect_eeg_sync_artifact(
            raw,
            freq_low=1,
            freq_high=50,
            channel_list=["CPz", "Pz", "Oz", "Fz", "Cz", "T7", "T8", "O1", "O2"],
            smooth_window=301,
            window_size_sec=2,
            plot=False,
            save_dir=None,
        )

        # Types are either valid or None when detection isn't possible on random noise
        self.assertTrue(isinstance(sync_channel, (str, type(None))))
        self.assertTrue(isinstance(sync_idx, (int, type(None))))
        self.assertTrue(isinstance(sync_time, (float, type(None))))
        self.assertTrue(isinstance(sync_result, (dict, type(None))))
        self.assertTrue(isinstance(sync_power, (np.ndarray, type(None))))

        # If detection succeeded, validate expected keys in the result dict
        if sync_result is not None:
            for key in ["onset_index", "onset_time", "channel", "type"]:
                self.assertIn(key, sync_result)

    def test_confirm_sync_selection(self):
        channel = "Cz"
        sync_time = 1.0
        result_type = "drop"
        magnitude = 1.0
        result = confirm_sync_selection(channel, sync_time, result_type, magnitude)
        self.assertIsInstance(result, bool)

    def test_save_sync_peak_info(self):
        eeg_file = "eeg_file.csv"
        dbs_file = "dbs_file.csv"
        eeg_peak_idx = 1
        eeg_peak_time = 1.0
        dbs_peak_idx = 1
        dbs_peak_time = 1.0

if __name__ == '__main__':
    unittest.main()
