"""dbs-eeg-sync: EEG-DBS synchronization toolkit."""

from .core import sync_run, SyncResult, SyncError
from .data_loader import load_eeg_data, open_json_file
from .synchronizer import cut_data_at_sync, synchronize_data

__version__ = "0.1.0"
__all__ = [
    "sync_run",
    "SyncResult", 
    "SyncError",
    "load_eeg_data",
    "open_json_file",
    "cut_data_at_sync",
    "synchronize_data",
]