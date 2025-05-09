import unittest
import mne
import pandas as pd
import numpy as np
from source.power_calculater import compute_eeg_power, compute_samplewise_eeg_power
from source.data_loader import load_eeg_data

class TestPowerCalculator(unittest.TestCase):
    def setUp(self):
        # Create a sample EEG data
        info = mne.create_info(ch_names=['POz'], sfreq=1000, ch_types=['eeg'])
        data = np.random.randn(1, 100000)
        self.raw = mne.io.RawArray(data, info)

    def test_compute_eeg_power(self):
        # Test compute_eeg_power with default parameters
        power, time_axis = compute_eeg_power(self.raw, 8, 12, duration_sec=10)
        self.assertEqual(power.shape, (10001, ))
        self.assertEqual(time_axis.shape, (10001,))

    def test_compute_samplewise_eeg_power(self):
        # Test compute_samplewise_eeg_power with default parameters
        power, time_axis = compute_samplewise_eeg_power(self.raw, 8, 12, duration_sec=10)
        self.assertEqual(power.shape, (10001,))
        self.assertEqual(time_axis.shape, (10001,))

if __name__ == '__main__':
    unittest.main()