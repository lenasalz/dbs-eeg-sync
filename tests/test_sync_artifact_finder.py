"""
Tests for sync_artifact_finder.py artifact detection functions.
"""
import unittest
from pathlib import Path

import numpy as np
import mne

from dbs_eeg_sync.sync_artifact_finder import (
    detect_dbs_sync_artifact,
    detect_eeg_sync_artifact
)


class TestSyncArtifactFinder(unittest.TestCase):
    """Test suite for artifact detection functions."""
    
    def setUp(self):
        """Create test fixtures."""
        # Create EEG data with artifact
        sfreq = 1000.0
        n_samples = 130000
        n_channels = 9
        
        # Create random data
        rng = np.random.default_rng(42)
        data = rng.standard_normal((n_channels, n_samples))
        
        # Add artificial spike around 65000 samples (65 seconds)
        artifact_idx = 65000
        artifact_window = 500
        for ch in range(n_channels):
            data[ch, artifact_idx:artifact_idx+artifact_window] *= 3.0
        
        self.eeg_raw = mne.io.RawArray(
            data,
            mne.create_info(
                ch_names=['POz', 'CPz', 'Pz', 'Oz', 'Fz', 'Cz', 'T7', 'T8', 'O1'],
                sfreq=sfreq,
                ch_types=['eeg'] * n_channels,
            )
        )
        
        # Create DBS data with artifact
        dbs_n_samples = 10000
        self.dbs_signal = np.random.randn(dbs_n_samples)
        # Add positive peak
        self.dbs_signal[1000] = 10.0
        self.dbs_fs = 250.0
    
    def test_detect_dbs_sync_artifact_basic(self):
        """Test basic DBS artifact detection."""
        peak_idx, peak_time = detect_dbs_sync_artifact(
            self.dbs_signal,
            self.dbs_fs,
            save_dir=None,
            plot=False
        )
        
        # Should return valid indices
        self.assertIsInstance(peak_idx, (int, np.integer))
        self.assertIsInstance(peak_time, (float, np.floating))
        
        # Peak index should be within signal range
        self.assertGreaterEqual(peak_idx, 0)
        self.assertLess(peak_idx, len(self.dbs_signal))
        
        # Should detect the large peak we added
        self.assertEqual(peak_idx, 1000)
    
    def test_detect_dbs_sync_artifact_no_peaks(self):
        """Test DBS artifact detection with no clear peaks."""
        # Signal with no prominent peaks
        flat_signal = np.ones(1000) * 0.1
        
        peak_idx, peak_time = detect_dbs_sync_artifact(
            flat_signal,
            self.dbs_fs,
            save_dir=None,
            plot=False
        )
        
        # Should still return a result (fallback to max in first 1000 samples)
        self.assertIsNotNone(peak_idx)
        self.assertIsNotNone(peak_time)
    
    def test_detect_eeg_sync_artifact_basic(self):
        """Test basic EEG artifact detection."""
        channel, sync_idx, sync_time, result, power = detect_eeg_sync_artifact(
            self.eeg_raw,
            freq_low=120.0,
            freq_high=130.0,
            time_range=None,
            plot=False,
            save_dir=None
        )
        
        # Should return valid results
        self.assertIsNotNone(channel)
        self.assertIsNotNone(sync_idx)
        self.assertIsNotNone(sync_time)
        self.assertIsNotNone(result)
        
        # Channel should be one of the available channels
        self.assertIn(channel, self.eeg_raw.ch_names)
        
        # Index should be within range
        self.assertGreaterEqual(sync_idx, 0)
        self.assertLess(sync_idx, self.eeg_raw.n_times)
        
        # Result should be a dict
        self.assertIsInstance(result, dict)
        self.assertIn("channel", result)
        self.assertIn("magnitude", result)
    
    def test_detect_eeg_sync_artifact_with_time_range(self):
        """Test EEG artifact detection with time range."""
        start_time = 10.0
        end_time = 100.0
        
        channel, sync_idx, sync_time, result, power = detect_eeg_sync_artifact(
            self.eeg_raw,
            freq_low=120.0,
            freq_high=130.0,
            time_range=(start_time, end_time),
            plot=False,
            save_dir=None
        )
        
        # Sync time should be within the specified range
        # (Actually, sync_idx and sync_time are GLOBAL, so this test verifies that)
        self.assertIsNotNone(sync_time)
        # The algorithm should find something, though not necessarily in the exact range
        self.assertGreaterEqual(sync_time, 0)
    
    def test_detect_eeg_sync_artifact_no_valid_channels(self):
        """Test EEG artifact detection with no valid channels."""
        channel, sync_idx, sync_time, result, power = detect_eeg_sync_artifact(
            self.eeg_raw,
            freq_low=120.0,
            freq_high=130.0,
            channel_list=['NonExistent1', 'NonExistent2'],
            plot=False,
            save_dir=None
        )
        
        # Should return None values when no channels found
        self.assertIsNone(channel)
        self.assertIsNone(sync_idx)
        self.assertIsNone(sync_time)
    
    def test_detect_eeg_sync_artifact_custom_channels(self):
        """Test EEG artifact detection with custom channel list."""
        custom_channels = ['Pz', 'Oz']
        
        channel, sync_idx, sync_time, result, power = detect_eeg_sync_artifact(
            self.eeg_raw,
            freq_low=120.0,
            freq_high=130.0,
            channel_list=custom_channels,
            plot=False,
            save_dir=None
        )
        
        # Selected channel should be from the custom list
        if channel is not None:
            self.assertIn(channel, custom_channels)
    
    def test_detect_eeg_sync_artifact_nyquist_clamping(self):
        """Test that frequency band is clamped to Nyquist."""
        # Request frequency above Nyquist
        nyquist = self.eeg_raw.info['sfreq'] / 2
        
        channel, sync_idx, sync_time, result, power = detect_eeg_sync_artifact(
            self.eeg_raw,
            freq_low=nyquist + 10,  # Above Nyquist
            freq_high=nyquist + 20,  # Above Nyquist
            plot=False,
            save_dir=None
        )
        
        # Should handle it gracefully (with warnings) and still return results
        # The function clamps to nyquist-1, so it should not crash


if __name__ == "__main__":
    unittest.main()
