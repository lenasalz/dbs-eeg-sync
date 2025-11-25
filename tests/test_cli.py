"""
Tests for cli.py command-line interface.
"""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import json

from dbs_eeg_sync.cli import (
    build_arg_parser,
    _load_config_file,
    _merge_config,
    _parse_time_range,
)


class TestCLI(unittest.TestCase):
    """Test suite for CLI functionality."""
    
    def test_build_arg_parser(self):
        """Test that argument parser is created successfully."""
        parser = build_arg_parser()
        self.assertIsNotNone(parser)
    
    def test_parse_time_range_valid(self):
        """Test valid time range parsing."""
        result = _parse_time_range("10,60")
        self.assertEqual(result, (10.0, 60.0))
        
        result = _parse_time_range("0,120.5")
        self.assertEqual(result, (0.0, 120.5))
    
    def test_parse_time_range_none(self):
        """Test None time range."""
        result = _parse_time_range(None)
        self.assertIsNone(result)
    
    def test_parse_time_range_invalid(self):
        """Test that invalid time range raises error."""
        import argparse
        
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_time_range("60,10")  # end < start
    
    def test_load_config_file_json(self):
        """Test loading JSON config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"sub_id": "S01", "block": "B1"}, f)
            temp_path = Path(f.name)
        
        try:
            config = _load_config_file(temp_path)
            self.assertEqual(config["sub_id"], "S01")
            self.assertEqual(config["block"], "B1")
        finally:
            temp_path.unlink()
    
    def test_load_config_file_yaml(self):
        """Test loading YAML config file (if pyyaml available)."""
        try:
            import yaml
            has_yaml = True
        except ImportError:
            has_yaml = False
        
        if not has_yaml:
            self.skipTest("PyYAML not installed")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("sub_id: S01\nblock: B1\n")
            temp_path = Path(f.name)
        
        try:
            config = _load_config_file(temp_path)
            self.assertEqual(config["sub_id"], "S01")
            self.assertEqual(config["block"], "B1")
        finally:
            temp_path.unlink()
    
    def test_load_config_file_not_found(self):
        """Test that FileNotFoundError is raised for missing config."""
        with self.assertRaises(FileNotFoundError):
            _load_config_file(Path("nonexistent.json"))
    
    def test_merge_config(self):
        """Test configuration merging with correct precedence."""
        defaults = {"a": 1, "b": 2, "c": 3}
        config = {"b": 20, "c": 30}
        flags = {"c": 300, "d": None}
        
        result = _merge_config(defaults, config, flags)
        
        # Precedence: defaults < config < flags (when not None)
        self.assertEqual(result["a"], 1)      # from defaults
        self.assertEqual(result["b"], 20)     # from config
        self.assertEqual(result["c"], 300)    # from flags
        self.assertIsNone(result.get("d"))    # None value ignored


class TestCLIIntegration(unittest.TestCase):
    """Integration tests for CLI (requires example data)."""
    
    def setUp(self):
        """Check if example data exists."""
        self.eeg_file = Path("data/eeg_example.set")
        self.dbs_file = Path("data/dbs_example.json")
        
        if not self.eeg_file.exists() or not self.dbs_file.exists():
            self.skipTest("Example data files not found")
    
    def test_cli_help(self):
        """Test that CLI help works."""
        parser = build_arg_parser()
        # This shouldn't raise an error
        help_text = parser.format_help()
        self.assertIn("synchronize", help_text.lower())


if __name__ == "__main__":
    unittest.main()

