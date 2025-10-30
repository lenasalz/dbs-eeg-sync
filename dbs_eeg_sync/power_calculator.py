"""
power_calculator.py — Sample-wise EEG Power Calculation
------------------------------------------------------
Contains routines for computing sample-wise band power of EEG data.
"""

from __future__ import annotations
import mne  
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def compute_samplewise_eeg_power(
    eeg_raw: mne.io.Raw,
    freq_low: int,
    freq_high: int,
    channel: str = 'POz',
    smoothing_sec: float | None = 0.5,
    smooth_window: int = 301,
    plot: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes sample-wise band power of EEG data in a specified frequency range, with optional smoothing. 
    The sample-wise power is computed by averaging the power of the EEG data in the specified frequency range.
    The resulting power trace is of the same length as the EEG data.

    Args:
        raw_eeg (mne.io.Raw): The raw EEG data.
        freq_low (int): Lower frequency bound.
        freq_high (int): Upper frequency bound.
        channel (str): Channel to analyze (default "POz").
        smoothing_sec (float): Smoothing window length in seconds (default 0.5s).
        plot (bool): Whether to plot the power trace.

    Returns:
        tuple[np.ndarray, np.ndarray]: (power_trace, time_axis)
    """
    
    # Copy, crop, and filter EEG
    _raw = eeg_raw.copy().pick(channel)
    fs = _raw.info["sfreq"]
    _raw.filter(l_freq=freq_low, h_freq=freq_high, verbose='ERROR')

    # Get data
    data = _raw.get_data().squeeze()  # shape (n_samples,)

    # Hilbert transform to get analytic signal
    analytic_signal = hilbert(data)
    power_trace = np.abs(analytic_signal) ** 2
    
    # Apply Savitzky-Golay smoothing
    if smooth_window and smooth_window > 1:
        # Ensure odd and less than signal length
        if smooth_window % 2 == 0:
            smooth_window += 1
        smooth_window = min(smooth_window, len(power_trace) - 1 if len(power_trace) % 2 == 0 else len(power_trace))
        power_trace_smoothed = savgol_filter(power_trace, window_length=smooth_window, polyorder=3)

    # Time axis
    time_axis = np.arange(len(power_trace)) / fs

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, power_trace, label=f'{freq_low}-{freq_high} Hz Power')
        plt.xlabel("Time (s)")
        plt.ylabel("Power")
        plt.title(f"Sample-wise Band Power ({channel}))")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if smooth_window and smooth_window > 1:
        return power_trace_smoothed, time_axis
    else:
        return power_trace, time_axis



if __name__ == '__main__':
    from pathlib import Path
    from dbs_eeg_sync.data_loader import load_eeg_data
    
    # Use relative path from repository root
    file_path = Path(__file__).parent.parent / "data" / "eeg_example.set"
    eeg_data, fs = load_eeg_data(file_path)
    power, time = compute_samplewise_eeg_power(eeg_data, 8, 12, channel="T8", plot=True)
    
    # Save power to csv in outputs folder
    output_path = Path(__file__).parent.parent / "outputs" / "outputData" / "eeg_power.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, power, delimiter=",")
    print(f"Power shape: {power.shape}, Time shape: {time.shape}")


    