#!/usr/bin/env python3
"""
Quick manual test for the GUI.
This is NOT a unittest - just a quick way to verify the GUI works.

Run: python tests/test_gui_quick.py
"""
import numpy as np
import matplotlib
matplotlib.use('QtAgg')

from dbs_eeg_sync.gui import manual_select_sync

# Create simple synthetic data
eeg_fs = 2000
t = np.linspace(0, 10, eeg_fs * 10)
eeg_signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))

# Add artifact at t=5s
eeg_signal[int(5 * eeg_fs):int(5 * eeg_fs) + 100] += 10

print("Opening GUI... Select a sync point and click 'Confirm selection'")

try:
    eeg_idx, eeg_time, dbs_idx, dbs_time = manual_select_sync(
        eeg_data=eeg_signal,
        eeg_fs=eeg_fs,
        dbs_data=None,
        dbs_fs=None,
        title="Quick GUI Test"
    )
    print(f"\n✓ Success! Selected: index={eeg_idx}, time={eeg_time:.3f}s")
except RuntimeError as e:
    print(f"\n✗ {e}")
