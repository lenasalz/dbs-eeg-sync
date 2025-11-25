# tests/test_core.py
"""
Tests for core.py synchronization functionality.
"""
import unittest
from pathlib import Path
import tempfile
import json

from dbs_eeg_sync.core import (
    sync_run, 
    SyncResult, 
    SyncError,
    ensure_output_dir,
    ensure_matplotlib_headless,
    save_run_metadata,
    result_to_dict,
)


class TestCore(unittest.TestCase):
    """Test suite for core synchronization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.eeg_file = Path("data/eeg_example.set")
        self.dbs_file = Path("data/dbs_example.json")
        
        # Skip tests if example files don't exist
        if not self.eeg_file.exists() or not self.dbs_file.exists():
            self.skipTest("Example data files not found")
    
    def test_sync_run_smoke(self):
        """Basic smoke test: sync_run completes without errors."""
        res = sync_run(
            sub_id="Sx", 
            block="B1", 
            eeg_file=self.eeg_file, 
            dbs_file=self.dbs_file
        )
        
        # Check that synchronization indices are valid
        self.assertGreaterEqual(res.eeg_sync_idx, 0)
        self.assertGreaterEqual(res.dbs_sync_idx, 0)
        
        # Check that synchronized data objects exist
        self.assertIsNotNone(res.synchronized_eeg)
        self.assertIsNotNone(res.synchronized_dbs)
        
        # Check that cropped data exists
        self.assertIsNotNone(res.cropped_eeg)
        self.assertIsNotNone(res.cropped_dbs)
    
    def test_sync_run_with_time_range(self):
        """Test sync_run with time_range cropping."""
        res = sync_run(
            sub_id="Sx",
            block="B1",
            eeg_file=self.eeg_file,
            dbs_file=self.dbs_file,
            time_range=(10.0, 30.0)
        )
        
        self.assertIsNotNone(res)
        self.assertEqual(res.time_range, (10.0, 30.0))
        self.assertGreaterEqual(res.eeg_sync_idx, 0)
    
    def test_sync_run_result_structure(self):
        """Test that SyncResult has all expected attributes."""
        res = sync_run(
            sub_id="test_sub",
            block="test_block",
            eeg_file=self.eeg_file,
            dbs_file=self.dbs_file
        )
        
        # Check required attributes
        self.assertEqual(res.sub_id, "test_sub")
        self.assertEqual(res.block, "test_block")
        self.assertEqual(res.eeg_file, self.eeg_file)
        self.assertEqual(res.dbs_file, self.dbs_file)
        
        # Check numeric attributes
        self.assertIsInstance(res.eeg_sync_idx, int)
        self.assertIsInstance(res.eeg_sync_s, float)
        self.assertIsInstance(res.dbs_sync_idx, int)
        self.assertIsInstance(res.dbs_sync_s, float)
        self.assertIsInstance(res.eeg_fs, float)
        self.assertIsInstance(res.dbs_fs, float)
        
        # Check that sampling rates are reasonable
        self.assertGreater(res.eeg_fs, 0)
        self.assertGreater(res.dbs_fs, 0)
        
        # Check metadata dict exists
        self.assertIsInstance(res.metadata, dict)
    
    def test_sync_run_invalid_eeg_file(self):
        """Test that FileNotFoundError is raised for missing EEG file."""
        with self.assertRaises(FileNotFoundError):
            sync_run(
                sub_id="Sx",
                block="B1",
                eeg_file=Path("nonexistent.set"),
                dbs_file=self.dbs_file
            )
    
    def test_sync_run_invalid_dbs_file(self):
        """Test that FileNotFoundError is raised for missing DBS file."""
        with self.assertRaises(FileNotFoundError):
            sync_run(
                sub_id="Sx",
                block="B1",
                eeg_file=self.eeg_file,
                dbs_file=Path("nonexistent.json")
            )
    
    def test_sync_run_invalid_time_range(self):
        """Test that ValueError is raised for invalid time_range."""
        with self.assertRaises(ValueError):
            sync_run(
                sub_id="Sx",
                block="B1",
                eeg_file=self.eeg_file,
                dbs_file=self.dbs_file,
                time_range=(100.0, 50.0)  # end < start (invalid)
            )
    
    def test_sync_run_no_auto_crop(self):
        """Test sync_run with auto_crop=False."""
        res = sync_run(
            sub_id="Sx",
            block="B1",
            eeg_file=self.eeg_file,
            dbs_file=self.dbs_file,
            auto_crop=False
        )
        
        # Should still return valid indices
        self.assertGreaterEqual(res.eeg_sync_idx, 0)
        self.assertGreaterEqual(res.dbs_sync_idx, 0)
        
        # Data should exist (but not cropped)
        self.assertIsNotNone(res.synchronized_eeg)
        self.assertIsNotNone(res.synchronized_dbs)


class TestCoreUtilities(unittest.TestCase):
    """Test suite for core utility functions."""
    
    def test_ensure_output_dir(self):
        """Test that ensure_output_dir creates directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_output" / "nested"
            result = ensure_output_dir(test_path)
            
            self.assertTrue(result.exists())
            self.assertTrue(result.is_dir())
            self.assertEqual(result, test_path)
    
    def test_ensure_output_dir_existing(self):
        """Test that ensure_output_dir handles existing directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir)
            result = ensure_output_dir(test_path)
            
            # Should not raise error for existing dir
            self.assertTrue(result.exists())
    
    def test_ensure_matplotlib_headless(self):
        """Test matplotlib backend configuration."""
        # Should complete without error
        result = ensure_matplotlib_headless(headless=True)
        
        # Result indicates whether backend was set
        self.assertIsInstance(result, bool)
    
    def test_result_to_dict(self):
        """Test SyncResult serialization to dict."""
        # Create a minimal mock result
        from dbs_eeg_sync.core import SyncResult
        
        res = SyncResult(
            sub_id="S01",
            block="B1",
            eeg_file=Path("eeg.set"),
            dbs_file=Path("dbs.json"),
            eeg_sync_idx=100,
            eeg_sync_s=0.5,
            dbs_sync_idx=50,
            dbs_sync_s=0.2,
            eeg_fs=200.0,
            dbs_fs=250.0,
            synchronized_eeg=None,
            synchronized_dbs=None,
            cropped_eeg=None,
            cropped_dbs=None,
            channel="Cz",
            artifact_kind="spike",
            artifact_magnitude=1.5,
            time_range=(0.0, 10.0),
            metadata={"test": "value"}
        )
        
        result_dict = result_to_dict(res)
        
        # Check that dict contains expected keys
        self.assertEqual(result_dict["sub_id"], "S01")
        self.assertEqual(result_dict["block"], "B1")
        self.assertEqual(result_dict["eeg_sync_idx"], 100)
        self.assertEqual(result_dict["dbs_sync_idx"], 50)
        self.assertEqual(result_dict["channel"], "Cz")
        self.assertEqual(result_dict["artifact_kind"], "spike")
    
    def test_save_run_metadata(self):
        """Test metadata JSON saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from dbs_eeg_sync.core import SyncResult
            
            res = SyncResult(
                sub_id="S01",
                block="B1",
                eeg_file=Path("eeg.set"),
                dbs_file=Path("dbs.json"),
                eeg_sync_idx=100,
                eeg_sync_s=0.5,
                dbs_sync_idx=50,
                dbs_sync_s=0.2,
                eeg_fs=200.0,
                dbs_fs=250.0,
                synchronized_eeg=None,
                synchronized_dbs=None,
                cropped_eeg=None,
                cropped_dbs=None,
                channel="Cz",
                artifact_kind="spike",
                artifact_magnitude=1.5,
                time_range=(0.0, 10.0),
                metadata={"test": "value"}
            )
            
            output_path = save_run_metadata(res, Path(tmpdir))
            
            # Check file exists
            self.assertTrue(output_path.exists())
            
            # Check it's valid JSON
            with output_path.open() as f:
                data = json.load(f)
            
            self.assertEqual(data["sub_id"], "S01")
            self.assertEqual(data["block"], "B1")
            self.assertIn("versions", data)
            self.assertIn("git", data)


if __name__ == "__main__":
    unittest.main()
