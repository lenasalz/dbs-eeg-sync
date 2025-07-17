import unittest
import mne
import pandas as pd
import numpy as np
import numbers

import sys
import os

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source.power_calculater import compute_samplewise_eeg_power
from source.sync_artefact_finder import find_eeg_peak, find_dbs_peak, detect_eeg_drop_onset_window, detect_eeg_sync_window, detect_sync_from_eeg, confirm_sync_selection


class TestArtefactFinderFunctions(unittest.TestCase):

    def test_find_eeg_peak(self):
        raw = mne.io.RawArray(np.random.randn(9, 130000), mne.create_info(ch_names=['POz', 'CPz', 'Pz', 'Oz', 'Fz', 'Cz', 'T7', 'T8', 'O1'], sfreq=1000, ch_types=['eeg'] * 9))
        power = compute_samplewise_eeg_power(raw, freq_low=110, freq_high=130)
        peak_idx, peak_time = find_eeg_peak(power[0], raw.info['sfreq'], save_dir=None)
        self.assertIsInstance(peak_idx, int)
        self.assertIsInstance(peak_time, float)

    def test_find_dbs_peak(self):
        data = pd.DataFrame({'TimeDomainData': np.random.randn(10000), 'SampleRateInHz': [1000]*10000})
        peak_idx, peak_time = find_dbs_peak(data.iloc[:, 0], dbs_fs=1000)
        self.assertIsInstance(peak_idx, (numbers.Integral, type(None)))  # index is integer
        self.assertIsInstance(peak_time, (numbers.Real, type(None)))    # time is float

    def test_detect_eeg_drop_onset_window(self):
        raw = mne.io.RawArray(np.random.randn(9, 130000), mne.create_info(ch_names=['POz', 'CPz', 'Pz', 'Oz', 'Fz', 'Cz', 'T7', 'T8', 'O1'], sfreq=1000, ch_types=['eeg'] * 9))
        power = compute_samplewise_eeg_power(raw, freq_low=110, freq_high=130)
        drop_onset_idx, drop_onset_time, smoothed = detect_eeg_drop_onset_window(power, raw.info['sfreq'], smooth_window=301, window_size_sec=2, plot=False, save_dir=None)
        self.assertTrue(isinstance(drop_onset_idx, (int, type(None))))
        self.assertTrue(isinstance(drop_onset_time, (int, type(None))))
        self.assertIsInstance(smoothed, np.ndarray)

    def test_detect_eeg_sync_window(self):
        raw = mne.io.RawArray(np.random.randn(9, 130000), mne.create_info(ch_names=['POz', 'CPz', 'Pz', 'Oz', 'Fz', 'Cz', 'T7', 'T8', 'O1'], sfreq=1000, ch_types=['eeg'] * 9))
        power = compute_samplewise_eeg_power(raw, freq_low=110, freq_high=130)
        result, smoothed = detect_eeg_sync_window(power, raw.info['sfreq'], smooth_window=301, window_size_sec=2, plot=False, save_dir=None)
        self.assertIsInstance(result, dict)
        self.assertIsInstance(smoothed, np.ndarray)

    def test_detect_sync_from_eeg(self):
        raw = mne.io.RawArray(np.random.randn(9, 130000), mne.create_info(ch_names=['POz', 'CPz', 'Pz', 'Oz', 'Fz', 'Cz', 'T7', 'T8', 'O1'], sfreq=1000, ch_types=['eeg'] * 9))
        sync_channel, sync_idx, sync_time, sync_result, sync_smoothed = detect_sync_from_eeg(raw, freq_low=1, freq_high=50, channel_list=["CPz", "Pz", "Oz", "Fz", "Cz", "T7", "T8", "O1", "O2"], smooth_window=301, window_size_sec=2, plot=False, save_dir=None)
        self.assertIsInstance(sync_channel, str)
        self.assertIsInstance(sync_idx, int)
        self.assertIsInstance(sync_time, float)
        self.assertIsInstance(sync_result, dict)
        self.assertIsInstance(sync_smoothed, np.ndarray)

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
