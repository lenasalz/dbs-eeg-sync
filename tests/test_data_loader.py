import unittest
from unittest.mock import patch, mock_open
import pandas as pd
from source.data_loader import (
    load_eeg_data, 
    open_json_file, 
    select_recording, 
    read_time_domain_data, 
    read_lfp_data
)

class TestDataLoader(unittest.TestCase):
    
    @patch("os.path.exists", return_value=True)
    @patch("mne.io.read_raw_eeglab")
    def test_load_eeg_data_set(self, mock_read_raw_eeglab, mock_exists):
        mock_read_raw_eeglab.return_value = "mock_eeg_data"
        file_path = "test.set"
        result = load_eeg_data(file_path)
        self.assertEqual(result, "mock_eeg_data")
        mock_read_raw_eeglab.assert_called_once_with(file_path, preload=True)
    
    def test_load_eeg_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_eeg_data("nonexistent.set")
    
    def test_load_eeg_data_unsupported_format(self):
        with patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError):
                load_eeg_data("test.unsupported")
    
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    def test_open_json_file(self, mock_file):
        result = open_json_file("test.json")
        self.assertEqual(result, {"key": "value"})
        mock_file.assert_called_once_with("test.json")
    
    def test_select_block_no_blocks(self):
        with self.assertRaises(ValueError):
            select_recording({"BrainSenseTimeDomain": []})
    
    @patch("builtins.input", side_effect=["0"])
    def test_select_block_valid(self, mock_input):
        json_data = {"BrainSenseTimeDomain": [{"SampleRateInHz": 250}]} 
        result = select_recording(json_data)
        self.assertEqual(result, 0)
    
    @patch("builtins.input", side_effect=["abc", "3", "-1", "0"])  # Invalid -> Invalid -> Invalid -> Valid (0)
    def test_select_block_invalid_then_valid(self, mock_input):
        json_data = {"BrainSenseTimeDomain": [{}, {}]}  # Two blocks (0 and 1)
        result = select_recording(json_data)
        # Ensure the chosen block is within valid range
        self.assertTrue(0 <= result < len(json_data["BrainSenseTimeDomain"]))
    
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
