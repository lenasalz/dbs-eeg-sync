"""
Tests for synchronizer.py signal alignment and resampling functions.
"""
import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import mne

from dbs_eeg_sync.synchronizer import cut_data_at_sync, synchronize_data


class TestSynchronizer(unittest.TestCase):
    """Test suite for synchronizer functions."""
    
    def setUp(self):
        """Create test fixtures."""
        # Create simple test EEG data
        n_channels = 2
        n_samples = 10000
        sfreq = 1000.0
        
        data = np.random.randn(n_channels, n_samples)
        info = mne.create_info(
            ch_names=['Ch1', 'Ch2'], 
            sfreq=sfreq, 
            ch_types=['eeg', 'eeg']
        )
        self.eeg_raw = mne.io.RawArray(data, info)
        
        # Create simple test DBS data
        self.dbs_df = pd.DataFrame({
            'TimeDomainData': np.random.randn(n_samples),
            'SampleRateInHz': [sfreq] * n_samples
        })
    
    def test_cut_data_at_sync_basic(self):
        """Test basic data cropping at sync indices."""
        eeg_sync_idx = 500
        dbs_sync_idx = 500
        
        cropped_eeg, cropped_dbs = cut_data_at_sync(
            self.eeg_raw, 
            self.dbs_df, 
            dbs_sync_idx, 
            eeg_sync_idx
        )
        
        # Check that data was cropped
        self.assertIsInstance(cropped_eeg, mne.io.BaseRaw)
        self.assertIsInstance(cropped_dbs, pd.DataFrame)
        
        # Check that cropped data is shorter
        self.assertLess(cropped_eeg.n_times, self.eeg_raw.n_times)
        self.assertLess(len(cropped_dbs), len(self.dbs_df))
    
    def test_cut_data_at_sync_zero_index(self):
        """Test cropping with zero sync index (no crop)."""
        cropped_eeg, cropped_dbs = cut_data_at_sync(
            self.eeg_raw, 
            self.dbs_df, 
            dbs_sync_idx=0, 
            eeg_sync_idx=0
        )
        
        # Should be same length (or very close)
        self.assertEqual(cropped_eeg.n_times, self.eeg_raw.n_times)
        self.assertEqual(len(cropped_dbs), len(self.dbs_df))
    
    def test_cut_data_at_sync_invalid_index(self):
        """Test that invalid sync index raises ValueError."""
        with self.assertRaises(ValueError):
            cut_data_at_sync(
                self.eeg_raw, 
                self.dbs_df, 
                dbs_sync_idx=99999,  # Out of range
                eeg_sync_idx=500
            )
    
    def test_synchronize_data_no_resample(self):
        """Test synchronization without resampling."""
        cropped_eeg, cropped_dbs = cut_data_at_sync(
            self.eeg_raw, self.dbs_df, 
            dbs_sync_idx=100, 
            eeg_sync_idx=100
        )
        
        sync_eeg, sync_dbs = synchronize_data(
            cropped_eeg, 
            cropped_dbs, 
            resample_data=False,
            save_dir=None
        )
        
        # Should return data objects
        self.assertIsInstance(sync_eeg, mne.io.BaseRaw)
        self.assertIsInstance(sync_dbs, pd.DataFrame)
        
        # Sampling rate should be unchanged
        self.assertEqual(sync_eeg.info['sfreq'], self.eeg_raw.info['sfreq'])
    
    def test_synchronize_data_with_resample(self):
        """Test synchronization with resampling."""
        # Create data with different sampling rates
        eeg_fs = 1000.0
        dbs_fs = 800.0
        
        eeg_data = np.random.randn(2, 10000)
        eeg_info = mne.create_info(['Ch1', 'Ch2'], sfreq=eeg_fs, ch_types=['eeg', 'eeg'])
        eeg_raw = mne.io.RawArray(eeg_data, eeg_info)
        
        dbs_df = pd.DataFrame({
            'TimeDomainData': np.random.randn(8000),
            'SampleRateInHz': [dbs_fs] * 8000
        })
        
        sync_eeg, sync_dbs = synchronize_data(
            eeg_raw, 
            dbs_df, 
            resample_data=True,
            save_dir=None
        )
        
        # Should be resampled to min(eeg_fs, dbs_fs)
        target_fs = min(eeg_fs, dbs_fs)
        self.assertAlmostEqual(sync_eeg.info['sfreq'], target_fs, places=1)
        self.assertAlmostEqual(sync_dbs['SampleRateInHz'].iloc[0], target_fs, places=1)
    
    def test_synchronize_data_with_save_dir(self):
        """Test that synchronize_data creates plots when save_dir is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cropped_eeg, cropped_dbs = cut_data_at_sync(
                self.eeg_raw, self.dbs_df, 
                dbs_sync_idx=100, 
                eeg_sync_idx=100
            )
            
            sync_eeg, sync_dbs = synchronize_data(
                cropped_eeg, 
                cropped_dbs, 
                resample_data=False,
                save_dir=tmpdir,
                sub_id="test_sub",
                block="test_block"
            )
            
            # Check that plot was saved
            plot_files = list(Path(tmpdir).glob("*.png"))
            self.assertGreater(len(plot_files), 0, "Expected at least one plot file")


if __name__ == "__main__":
    unittest.main()
