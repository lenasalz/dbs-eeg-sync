import unittest
from source.synchronizer import find_eeg_peak, find_dbs_peak, crop_data, synchronize_data, save_synchronized_data
import mne
import pandas as pd
import numpy as np

class TestSynchronizerFunctions(unittest.TestCase):

    def test_find_eeg_peak(self):
        raw = mne.io.RawArray(np.random.randn(2, 10000), mne.create_info(ch_names=['eeg1', 'eeg2'], sfreq=1000, ch_types=['eeg', 'eeg']))
        peak = find_eeg_peak(raw, freq_low=1, freq_high=50, decim=10, duration_sec=9)
        self.assertIsInstance(peak, int)

    def test_find_dbs_peak(self):
        data = pd.DataFrame({'TimeDomainData': np.random.randn(10000), 'SampleRateInHz': [1000]*10000})
        peak = find_dbs_peak(data)
        self.assertTrue(isinstance(peak, (int, np.integer)))

    def test_crop_data(self):
        raw = mne.io.RawArray(np.random.randn(2, 10000), mne.create_info(ch_names=['eeg1', 'eeg2'], sfreq=1000, ch_types=['eeg', 'eeg']))
        dbs_data = pd.DataFrame({'TimeDomainData': np.random.randn(10000), 'SampleRateInHz': [1000]*10000})
        eeg_cropped, dbs_cropped = crop_data(raw, dbs_data, 500, 1000)
        self.assertTrue(len(eeg_cropped.times) > 0)
        self.assertTrue(len(dbs_cropped) > 0)

    def test_synchronize_data(self):
        raw = mne.io.RawArray(np.random.randn(2, 10000), mne.create_info(ch_names=['eeg1', 'eeg2'], sfreq=1000, ch_types=['eeg', 'eeg']))
        dbs_data = pd.DataFrame({'TimeDomainData': np.random.randn(10000), 'SampleRateInHz': [1000]*10000})
        eeg_sync, dbs_sync = synchronize_data(raw, dbs_data)
        self.assertTrue(eeg_sync.get_data().shape[1] > 0)
        self.assertTrue(len(dbs_sync) > 0)

    def test_save_synchronized_data(self):
        raw = mne.io.RawArray(np.random.randn(2, 10000), mne.create_info(ch_names=['eeg1', 'eeg2'], sfreq=1000, ch_types=['eeg', 'eeg']))
        dbs_data = pd.DataFrame({'TimeDomainData': np.random.randn(10000), 'SampleRateInHz': [1000]*10000})
        save_synchronized_data(raw, dbs_data, output_dir="data")

if __name__ == '__main__':
    unittest.main()
