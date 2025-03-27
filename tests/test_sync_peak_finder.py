import unittest
import mne
import pandas as pd
import numpy as np
from source.sync_peaks_finder import find_eeg_peak, find_dbs_peak


class TestPeakFinderFunctions(unittest.TestCase):

    def test_find_eeg_peak(self):
        raw = mne.io.RawArray(np.random.randn(2, 10000), mne.create_info(ch_names=['eeg1', 'eeg2'], sfreq=1000, ch_types=['eeg', 'eeg']))
        peak = find_eeg_peak(raw, freq_low=1, freq_high=50, decim=10, duration_sec=9)
        self.assertIsInstance(peak, int)

    def test_find_dbs_peak(self):
        data = pd.DataFrame({'TimeDomainData': np.random.randn(10000), 'SampleRateInHz': [1000]*10000})
        peak = find_dbs_peak(data)
        self.assertTrue(isinstance(peak, (int, np.integer)))

if __name__ == '__main__':
    unittest.main()
