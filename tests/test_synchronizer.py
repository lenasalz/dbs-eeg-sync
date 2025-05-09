import unittest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from source.synchronizer import synchronize_data, save_synchronized_data, crop_data, save_sync_info
import mne
import pandas as pd
import numpy as np
import os
class TestSynchronizerFunctions(unittest.TestCase):

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

    def test_save_sync_info(self):
        save_sync_info(sub_id="sub1", block="block1", eeg_file="eeg_file.fif", dbs_file="dbs_file.csv", eeg_peak_idx=100, eeg_peak_time=1.0, dbs_peak_idx=200, dbs_peak_time=2.0)
        self.assertTrue(os.path.exists("outputs/sync_info.csv"))
        os.remove("outputs/sync_info.csv")


if __name__ == '__main__':
    unittest.main()
