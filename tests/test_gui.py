#!/usr/bin/env python3
"""
Unit tests for the manual_select_sync GUI.

These tests require:
- PyQt5 or PyQt6 installed
- A display/X11 available
- User interaction to confirm selections

To run: pytest tests/test_gui.py -v
Or mark as interactive only: pytest -m "not gui"
"""
import unittest
import numpy as np
import pytest

# Try to set Qt backend; skip tests if it fails
try:
    import matplotlib
    matplotlib.use('QtAgg')
    from dbs_eeg_sync.gui import manual_select_sync
    GUI_AVAILABLE = True
except (ImportError, RuntimeError):
    GUI_AVAILABLE = False


@pytest.mark.gui
@pytest.mark.skipif(not GUI_AVAILABLE, reason="GUI dependencies not available")
class TestManualSelectSync(unittest.TestCase):
    """Test suite for manual_select_sync GUI."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
    
    @pytest.mark.interactive
    def test_eeg_only(self):
        """Test with EEG data only."""
        # Create synthetic EEG data with a clear artifact
        eeg_fs = 2000  # Hz
        duration = 10  # seconds
        n_samples = int(eeg_fs * duration)
        
        # Create signal with artifact at t=5s
        t = np.linspace(0, duration, n_samples)
        eeg_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(n_samples)
        
        # Add artifact spike at 5 seconds
        artifact_idx = int(5 * eeg_fs)
        eeg_signal[artifact_idx:artifact_idx + 100] += 10
        
        try:
            eeg_idx, eeg_time, dbs_idx, dbs_time = manual_select_sync(
                eeg_data=eeg_signal,
                eeg_fs=eeg_fs,
                dbs_data=None,
                dbs_fs=None,
                title="Test: EEG Only"
            )
            
            # Assertions
            self.assertIsInstance(eeg_idx, int, "EEG index should be an integer")
            self.assertIsInstance(eeg_time, float, "EEG time should be a float")
            self.assertIsNone(dbs_idx, "DBS index should be None when no DBS data provided")
            self.assertIsNone(dbs_time, "DBS time should be None when no DBS data provided")
            self.assertGreaterEqual(eeg_idx, 0, "EEG index should be non-negative")
            self.assertLess(eeg_idx, n_samples, "EEG index should be within signal bounds")
            self.assertAlmostEqual(eeg_time, eeg_idx / eeg_fs, places=5, 
                                   msg="EEG time should match index/fs")
            
        except RuntimeError as e:
            if "cancelled" in str(e).lower():
                self.skipTest("User cancelled the selection")
            raise

    @pytest.mark.interactive
    def test_eeg_and_dbs(self):
        """Test with both EEG and DBS data."""
        # Create synthetic EEG data
        eeg_fs = 2000
        duration = 10
        n_eeg = int(eeg_fs * duration)
        t_eeg = np.linspace(0, duration, n_eeg)
        eeg_signal = np.sin(2 * np.pi * 10 * t_eeg) + 0.5 * np.random.randn(n_eeg)
        eeg_signal[int(5 * eeg_fs):int(5 * eeg_fs) + 100] += 10
        
        # Create synthetic DBS data
        dbs_fs = 250
        n_dbs = int(dbs_fs * duration)
        t_dbs = np.linspace(0, duration, n_dbs)
        dbs_signal = 0.3 * np.sin(2 * np.pi * 20 * t_dbs) + 0.1 * np.random.randn(n_dbs)
        dbs_signal[int(5 * dbs_fs):int(5 * dbs_fs) + 10] += 5
        
        try:
            eeg_idx, eeg_time, dbs_idx, dbs_time = manual_select_sync(
                eeg_data=eeg_signal,
                eeg_fs=eeg_fs,
                dbs_data=dbs_signal,
                dbs_fs=dbs_fs,
                title="Test: EEG + DBS"
            )
            
            # Assertions for EEG
            self.assertIsInstance(eeg_idx, int, "EEG index should be an integer")
            self.assertIsInstance(eeg_time, float, "EEG time should be a float")
            self.assertGreaterEqual(eeg_idx, 0, "EEG index should be non-negative")
            self.assertLess(eeg_idx, n_eeg, "EEG index should be within signal bounds")
            self.assertAlmostEqual(eeg_time, eeg_idx / eeg_fs, places=5)
            
            # Assertions for DBS
            self.assertIsInstance(dbs_idx, int, "DBS index should be an integer")
            self.assertIsInstance(dbs_time, float, "DBS time should be a float")
            self.assertGreaterEqual(dbs_idx, 0, "DBS index should be non-negative")
            self.assertLess(dbs_idx, n_dbs, "DBS index should be within signal bounds")
            self.assertAlmostEqual(dbs_time, dbs_idx / dbs_fs, places=5)
            
        except RuntimeError as e:
            if "cancelled" in str(e).lower():
                self.skipTest("User cancelled the selection")
            raise

    @pytest.mark.interactive
    def test_with_mne_data(self):
        """Test with actual MNE Raw object (if available)."""
        try:
            import mne
        except ImportError:
            self.skipTest("MNE not available")
        
        # Create synthetic MNE Raw object
        n_channels = 3
        sampling_freq = 2000
        duration = 10  # seconds
        n_times = int(sampling_freq * duration)
        
        # Create data with artifact
        data = np.random.randn(n_channels, n_times) * 1e-6  # Convert to volts
        # Add artifact at t=5s
        artifact_idx = int(5 * sampling_freq)
        data[:, artifact_idx:artifact_idx + 100] += 5e-6
        
        # Create MNE info and Raw object
        ch_names = ['Cz', 'Fz', 'Pz']
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sampling_freq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        
        try:
            eeg_idx, eeg_time, dbs_idx, dbs_time = manual_select_sync(
                eeg_data=raw,
                eeg_fs=sampling_freq,
                dbs_data=None,
                dbs_fs=None,
                title="Test: MNE Raw Object",
                freq_low=120,
                freq_high=130,
                channel='Cz'
            )
            
            # Assertions
            self.assertIsInstance(eeg_idx, int, "EEG index should be an integer")
            self.assertIsInstance(eeg_time, float, "EEG time should be a float")
            self.assertIsNone(dbs_idx, "DBS index should be None")
            self.assertIsNone(dbs_time, "DBS time should be None")
            self.assertGreaterEqual(eeg_idx, 0, "EEG index should be non-negative")
            self.assertLess(eeg_idx, n_times, "EEG index should be within signal bounds")
            
        except RuntimeError as e:
            if "cancelled" in str(e).lower():
                self.skipTest("User cancelled the selection")
            raise
    
    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        # Test missing both signals
        with self.assertRaises(ValueError, msg="Should raise error when both signals are None"):
            manual_select_sync(None, None, None, None, "Test")
        
        # Test EEG without fs
        eeg_signal = np.random.randn(1000)
        with self.assertRaises(ValueError, msg="Should raise error when eeg_fs is missing"):
            manual_select_sync(eeg_signal, None, None, None, "Test")
        
        # Test DBS without fs
        dbs_signal = np.random.randn(1000)
        with self.assertRaises(ValueError, msg="Should raise error when dbs_fs is missing"):
            manual_select_sync(None, None, dbs_signal, None, "Test")


if __name__ == "__main__":
    # Run with unittest
    unittest.main()

