#!/usr/bin/env python3
"""
Quick manual test for the GUI.
This is NOT a pytest unittest - just a quick way to verify the GUI works.
It opens an interactive window, so it must be run directly (not under pytest)
and requires a display plus PyQt6/PyQt5:

    python tests/test_gui_quick.py

All execution is guarded behind ``if __name__ == "__main__"`` so that pytest
can import this file during collection without selecting a Qt backend or
opening a window (which would otherwise fail on headless/CI machines).
"""


def main() -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("QtAgg")

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
            title="Quick GUI Test",
        )
        print(f"\n✓ Success! Selected: index={eeg_idx}, time={eeg_time:.3f}s")
    except RuntimeError as e:
        print(f"\n✗ {e}")


if __name__ == "__main__":
    main()
