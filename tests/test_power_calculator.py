"""
Tests for power_calculator.py band-power computation functions.
"""
import unittest

import numpy as np
import mne

from dbs_eeg_sync.power_calculator import compute_samplewise_eeg_power


class TestPowerCalculator(unittest.TestCase):
    """Test suite for power calculation functions."""
    
    def setUp(self):
        """Create test fixtures."""
        # Create sample EEG data with known frequency content
        sfreq = 1000.0
        n_samples = 10000
        t = np.arange(n_samples) / sfreq
        
        # Create signal with 10 Hz component
        signal = np.sin(2 * np.pi * 10 * t)
        
        info = mne.create_info(ch_names=['POz'], sfreq=sfreq, ch_types=['eeg'])
        data = signal.reshape(1, -1)
        self.raw = mne.io.RawArray(data, info)
    
    def test_compute_samplewise_eeg_power_basic(self):
        """Test basic power computation."""
        power, time_axis = compute_samplewise_eeg_power(
            self.raw, 
            freq_low=8, 
            freq_high=12,
            channel='POz'
        )
        
        # Check output shapes
        self.assertEqual(power.shape[0], self.raw.n_times)
        self.assertEqual(time_axis.shape[0], self.raw.n_times)
        
        # Check that power values are positive
        self.assertTrue(np.all(power >= 0))
        
        # Check that time axis is monotonic
        self.assertTrue(np.all(np.diff(time_axis) >= 0))
    
    def test_compute_samplewise_eeg_power_different_bands(self):
        """Test power computation with different frequency bands."""
        # Alpha band
        power_alpha, _ = compute_samplewise_eeg_power(
            self.raw, freq_low=8, freq_high=12, channel='POz'
        )
        
        # Beta band
        power_beta, _ = compute_samplewise_eeg_power(
            self.raw, freq_low=13, freq_high=30, channel='POz'
        )
        
        # Alpha should have more power (our signal is at 10 Hz)
        self.assertGreater(np.mean(power_alpha), np.mean(power_beta))
    
    def test_compute_samplewise_eeg_power_with_smoothing(self):
        """Test power computation with smoothing."""
        power_smoothed, _ = compute_samplewise_eeg_power(
            self.raw, 
            freq_low=8, 
            freq_high=12,
            channel='POz',
            smooth_window=301
        )
        
        power_unsmoothed, _ = compute_samplewise_eeg_power(
            self.raw, 
            freq_low=8, 
            freq_high=12,
            channel='POz',
            smooth_window=1  # Effectively no smoothing
        )
        
        # Smoothed should have less variance
        self.assertLess(np.std(power_smoothed), np.std(power_unsmoothed))
    
    def test_compute_samplewise_eeg_power_invalid_channel(self):
        """Test that invalid channel raises error."""
        with self.assertRaises(Exception):
            compute_samplewise_eeg_power(
                self.raw, 
                freq_low=8, 
                freq_high=12,
                channel='NonExistentChannel'
            )
    
    def test_compute_samplewise_eeg_power_multichannel(self):
        """Test with multi-channel data."""
        # Create multi-channel data
        sfreq = 1000.0
        n_samples = 10000
        n_channels = 3
        
        data = np.random.randn(n_channels, n_samples)
        info = mne.create_info(
            ch_names=['Ch1', 'Ch2', 'Ch3'], 
            sfreq=sfreq, 
            ch_types=['eeg'] * n_channels
        )
        raw_multi = mne.io.RawArray(data, info)
        
        # Should work for each channel
        for ch in ['Ch1', 'Ch2', 'Ch3']:
            power, time_axis = compute_samplewise_eeg_power(
                raw_multi, 
                freq_low=8, 
                freq_high=12,
                channel=ch
            )
            self.assertEqual(power.shape[0], n_samples)


if __name__ == '__main__':
    unittest.main()
