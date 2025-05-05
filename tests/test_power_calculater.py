import unittest
import mne
import pandas as pd
import numpy as np
from source.power_calculater import compute_eeg_power, compute_samplewise_eeg_power


class TestPowerCalculator(unittest.TestCase):
    def setUp(self):
        # Create a sample EEG data
        info = mne.create_info(ch_names=['eeg1', 'eeg2'], sfreq=1000, ch_types=['eeg', 'eeg'])
        data = np.random.randn(2, 10000)
        self.raw = mne.io.RawArray(data, info)

    def test_compute_eeg_power(self):
        # Test compute_eeg_power with default parameters
        power, time_axis = compute_eeg_power(self.raw)
        self.assertEqual(power.shape, (1, 10000))
        self.assertEqual(time_axis.shape, (10000,))

    def test_compute_samplewise_eeg_power(self):
        # Test compute_samplewise_eeg_power with default parameters
        power, time_axis = compute_samplewise_eeg_power(self.raw)
        self.assertEqual(power.shape, (2, 10000))
        self.assertEqual(time_axis.shape, (10000,))

if __name__ == '__main__':
    unittest.main()