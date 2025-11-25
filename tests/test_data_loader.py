import unittest
from unittest.mock import patch, mock_open
from pathlib import Path
import pandas as pd
from dbs_eeg_sync.data_loader import (
    load_eeg_data, 
    open_json_file, 
    select_recording, 
    read_time_domain_data, 
    read_lfp_data,
    dbs_artifact_settings
)

class TestDataLoader(unittest.TestCase):
    
    def test_load_eeg_data_set(self):
        """Test loading EEG file with .set format (requires example data)."""
        test_file = Path("data/eeg_example.set")
        
        if not test_file.exists():
            self.skipTest("Example data not available")
        
        raw, fs = load_eeg_data(test_file)
        self.assertIsNotNone(raw)
        self.assertGreater(fs, 0)
    
    def test_load_eeg_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_eeg_data("nonexistent.set")
    
    def test_load_eeg_data_unsupported_format(self):
        """Test that unsupported file format raises ValueError."""
        import tempfile
        
        # Create a temp file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".unsupported", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with self.assertRaises(ValueError):
                load_eeg_data(temp_path)
        finally:
            temp_path.unlink()
    
    def test_open_json_file(self):
        """Test JSON file loading."""
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"key": "value"}, f)
            temp_path = f.name
        
        try:
            result = open_json_file(temp_path)
            self.assertEqual(result, {"key": "value"})
        finally:
            Path(temp_path).unlink()
    
    def test_select_recording_no_blocks(self):
        """Test that ValueError is raised when no recordings are available."""
        with self.assertRaises(ValueError):
            select_recording({"BrainSenseTimeDomain": []})
    
    def test_select_recording_default_index(self):
        """Test that select_recording defaults to index 0 when no index is provided."""
        json_data = {"BrainSenseTimeDomain": [{"data": 1}, {"data": 2}]} 
        result = select_recording(json_data)
        self.assertEqual(result, 0)
    
    def test_select_recording_explicit_index(self):
        """Test select_recording with explicit index."""
        json_data = {"BrainSenseTimeDomain": [{"data": 1}, {"data": 2}, {"data": 3}]}
        result = select_recording(json_data, index=1)
        self.assertEqual(result, 1)
        
        result = select_recording(json_data, index=2)
        self.assertEqual(result, 2)
    
    def test_select_recording_invalid_index(self):
        """Test that ValueError is raised for out-of-range index."""
        json_data = {"BrainSenseTimeDomain": [{}, {}]}  # Two recordings (0 and 1)
        
        with self.assertRaises(ValueError):
            select_recording(json_data, index=5)  # Out of range
        
        with self.assertRaises(ValueError):
            select_recording(json_data, index=-1)  # Negative index
    
    def test_dbs_artifact_settings_defaults(self):
        """Test dbs_artifact_settings with default parameters."""
        fmin, fmax, tmin, tmax = dbs_artifact_settings()
        self.assertEqual(fmin, 120.0)
        self.assertEqual(fmax, 130.0)
        self.assertIsNone(tmin)
        self.assertIsNone(tmax)
    
    def test_dbs_artifact_settings_custom(self):
        """Test dbs_artifact_settings with custom parameters."""
        fmin, fmax, tmin, tmax = dbs_artifact_settings(
            freq_min=115.0, 
            freq_max=135.0, 
            tmin=10.0, 
            tmax=60.0
        )
        self.assertEqual(fmin, 115.0)
        self.assertEqual(fmax, 135.0)
        self.assertEqual(tmin, 10.0)
        self.assertEqual(tmax, 60.0)
    
    def test_read_time_domain_data(self):
        json_data = {"BrainSenseTimeDomain": [{"SampleRateInHz": 250, "TimeDomainData": [1, 2, 3]}]}
        df = read_time_domain_data(json_data, 0)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
    
    def test_read_lfp_data(self):
        for channel, expected_lead in [("LEFT", "Left"), ("RIGHT", "Right")]:
            json_data = {
                "BrainSenseLfp": [
                    {
                        "SampleRateInHz": 250,
                        "Channel": channel,
                        "LfpData": [{channel.capitalize(): {"LFP": 1}}, {channel.capitalize(): {"LFP": 2}}]
                    }
                ]
            }
            df, fs, lead = read_lfp_data(json_data, 0)

            self.assertEqual(fs, 250)
            self.assertEqual(lead, expected_lead)  # Dynamically assert based on input
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
    
if __name__ == "__main__":
    unittest.main()
