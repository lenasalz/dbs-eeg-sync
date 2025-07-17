# sync_peak_finder.py
# Functions for finding synchronization peaks in EEG and DBS data.
import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import plotly.graph_objects as go
from typing import Tuple, Optional, List

import sys
import os

# Add the local `source` directory to the import path
sys.path.insert(0, os.path.abspath("source"))
from power_calculater import compute_samplewise_eeg_power

import inspect


def find_eeg_peak(
        eeg_power,
        fs, 
        save_dir=None):
    """
    Finds the highest synchronization peak index in EEG data within a specified frequency band.

    Args:
        eeg_power (np.ndarray): The computed EEG power values.
        fs (int): Sampling frequency of the EEG signal.
        save_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
    Returns:
        int: Peak index in EEG samples.
        float: Peak time in seconds.
    """

    # Detect peaks
    peaks, peak_properties = find_peaks(eeg_power, height=np.mean(eeg_power) * 1.5)

    if len(peaks) > 0:
        # Select the peak with the highest power value
        highest_peak_idx = np.argmax(peak_properties["peak_heights"])
        peak_power_idx = peaks[highest_peak_idx]
    else:
        # If no peak is found, use the maximum value in the first 1000 samples as a fallback
        peak_power_idx = np.argmax(eeg_power[:1000])  

    eeg_peak_index_fs = int(peak_power_idx)
    eeg_peak_index_s = eeg_peak_index_fs / fs

    power_time_axis = np.arange(len(eeg_power)) / fs

    # Plot the detected peak
    plt.figure(figsize=(10, 5))
    plt.plot(power_time_axis, eeg_power, label="EEG Power")
    plt.axvline(power_time_axis[peak_power_idx], color='r', linestyle='--', label=f'Highest Peak @ {power_time_axis[peak_power_idx]:.2f} sec')
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.title('EEG: Power Over Time')
    plt.legend()
    
    if save_dir:
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/syncPeakEEG_{dat}.png")
        print(f"---\nPlot saved to {save_dir}/syncPeakEEG_{dat}.png")
    
    print("---\nPlease close the plot to continue.\n---")

    plt.show()  # Ensures the plot opens

    return eeg_peak_index_fs, eeg_peak_index_s


def find_dbs_peak(dbs_signal, dbs_fs, save_dir=None):
    """
    Finds the highest peak in DBS data in the positive direction only.

    Args:
        dbs_signal (np.ndarray): The DBS time series data.
        dbs_fs (int): Sampling frequency of the DBS.
        save_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
        log_file (str, optional): Log file path for saving detected peak info.

    Returns:
        int: Peak index in DBS samples.
        float: Peak time in seconds.
    """
    # Compute time axis
    dbs_time_axis = np.arange(len(dbs_signal)) / dbs_fs

    # Find peaks **only in the positive direction**
    peaks, _ = find_peaks(dbs_signal, height=0)  # Only positive peaks

    if len(peaks) > 0:
        # Select the **highest** positive peak
        dbs_peak_index_fs = peaks[np.argmax(dbs_signal[peaks])]
    else:
        # Fallback: Use max value in the first 1000 samples
        dbs_peak_index_fs = np.argmax(dbs_signal[:1000])

    dbs_peak_index_s = dbs_peak_index_fs / dbs_fs
    # Plot detected peak
    plt.figure(figsize=(10, 5))
    plt.plot(dbs_time_axis, dbs_signal, label="DBS Signal")
    plt.axvline(dbs_time_axis[dbs_peak_index_fs], color='r', linestyle='--', label=f'Peak @ {dbs_peak_index_s:.2f} sec | {dbs_peak_index_fs} samples')
    plt.xlabel('Time (s)')
    plt.ylabel('DBS Amplitude')
    plt.title('DBS Peak Detection')
    plt.legend()

    if save_dir:
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/syncPeakDBS_{dat}.png")
        print(f"---\nPlot saved to {save_dir}/syncPeakDBS_{dat}.png")
        print("---\nClose DBS sync plot to continue")
    
    plt.show()

    return dbs_peak_index_fs, dbs_peak_index_s


def detect_eeg_drop_onset_window(
    eeg_power,
    eeg_fs,
    smooth_window=301, 
    window_size_sec=2, 
    plot=False, 
    save_dir=None
) -> Tuple[Optional[int], Optional[float], np.ndarray]:
    """
    Detects all drops in EEG signal based on local window change.

    Args:
        eeg_power (np.ndarray): The computed EEG power values
        eeg_fs (int): Sampling frequency of the EEG signal
        smooth_window (int): Window size for smoothing the signal (default: 301)
        window_size_sec (int): Size of the window for drop detection in seconds (default: 2)
        plot (bool, optional): Indicate if the plot should be displayed (default: False)
        save_dir (str, optional): Directory to save the plot (default: None)

    Returns:
        drop_onset_idx (int): Index of the detected drop onset
        drop_onset_time (float): Time of the detected drop onset in seconds
        smoothed (np.ndarray): Smoothed EEG power signal
    """
    # --- Signal prep ---
    smoothed = savgol_filter(eeg_power, window_length=smooth_window, polyorder=3)

    # --- Drop detection ---
    window_size = int(window_size_sec * eeg_fs)

    min_diff = np.inf
    drop_onset_idx = None
    drop_onset_time = None
    for i in range(0, len(smoothed) - 2 * window_size):
        pre_window = smoothed[i : i + window_size]
        post_window = smoothed[i + window_size : i + 2 * window_size]
        diff = np.mean(post_window) - np.mean(pre_window)

        if diff < min_diff:
            min_diff = diff
            drop_onset_idx = i + window_size
            drop_onset_time = drop_onset_idx / eeg_fs

    if drop_onset_idx is None:
        print("⚠️ No clear drop detected. Consider adjusting parameters.")
        return None, None, smoothed 
    
    # --- Plotting ---

    if plot and drop_onset_idx is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=smoothed,
            mode="lines",
            name="Smoothed Power Band Sum"
        ))

        fig.add_vline(
            x=drop_onset_idx,
            line=dict(color="red", dash="dash"),
        )

        fig.add_annotation(
            x=drop_onset_idx - drop_onset_idx * 0.1,
            y=max(smoothed),
            text=f"Drop Idx: {drop_onset_idx} | Drop Time: {drop_onset_time:.2f}s",
            showarrow=False,
            font=dict(color="red")
        )

        fig.update_layout(
            title="EEG Drop Onset Detection (Windowed)",
            xaxis_title="Samples",
            yaxis_title="Power",
            legend=dict(x=0.01, y=0.99),
        )

        fig.show()

        # Save if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.write_image(os.path.join(save_dir, "drop_onset_detection_windowed.png"))
            
    return drop_onset_idx, drop_onset_time, smoothed


def detect_eeg_sync_window(
    eeg_power,
    eeg_fs,
    smooth_window=301,
    window_size_sec=2,
    plot=False,
    save_dir=None
) -> Tuple[dict, np.ndarray]:
    """
    Detects the steepest change in EEG signal based on local window change.
    Args:
        eeg_power (np.ndarray): The computed EEG power values
        eeg_fs (int): Sampling frequency of the EEG signal
        smooth_window (int): Window size for smoothing the signal (default: 301)
        window_size_sec (int): Size of the window for change detection in seconds (default: 2)
        plot (bool, optional): Indicate if the plot should be displayed (default: False)
        save_dir (str, optional): Directory to save the plot (default: None)
    Returns:
        result (dict): Dictionary containing the type of change, index, time, and magnitude
        smoothed (np.ndarray): Smoothed EEG power signal
    """
    # --- Signal prep ---

    smoothed = savgol_filter(eeg_power, window_length=smooth_window, polyorder=3)
    window_size = int(window_size_sec * eeg_fs)

    min_diff = np.inf
    max_diff = -np.inf
    drop_onset_idx = None
    spike_onset_idx = None

    for i in range(0, len(smoothed) - 2 * window_size):
        pre = smoothed[i : i + window_size]
        post = smoothed[i + window_size : i + 2 * window_size]
        diff = np.mean(post) - np.mean(pre)

        if diff < min_diff:
            min_diff = diff
            drop_onset_idx = i + window_size

        if diff > max_diff:
            max_diff = diff
            spike_onset_idx = i + window_size

    # Pick steeper one
    if abs(min_diff) >= abs(max_diff):
        result = {
            "type": "drop",
            "index": drop_onset_idx,
            "time": drop_onset_idx / eeg_fs if drop_onset_idx else None,
            "magnitude": min_diff,
        }
    else:
        result = {
            "type": "spike",
            "index": spike_onset_idx,
            "time": spike_onset_idx / eeg_fs if spike_onset_idx else None,
            "magnitude": max_diff,
        }

    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=smoothed, mode="lines", name="Smoothed Power"))
        fig.add_vline(
            x=result["index"], 
            line=dict(color="red" if result["type"] == "drop" else "green", dash="dash")
        )
        fig.update_layout(title="EEG Change Detection (Steepest)", xaxis_title="Samples", yaxis_title="Power")
        fig.show()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.write_image(os.path.join(save_dir, "change_onset_detection_steepest.png"))

    return result, smoothed


def detect_sync_from_eeg(
    eeg_raw: mne.io.Raw,
    freq_low: float,
    freq_high: float,
    time_range: Tuple[float, float] = (0, 120),
    eeg_fs: Optional[int] = None,
    channel_list: list = ['Pz', 'Oz', 'Fz', 'Cz', 'T7', 'T8', 'O1', 'O2'],
    smooth_window: int = 301,
    window_size_sec: int = 2,
    plot: bool = False,
    save_dir: Optional[str] = None,
) -> Tuple[
    Optional[str],
    List[int],
    List[float],
    List[dict],
    Optional[np.ndarray],
]:
    if eeg_fs is None:
        eeg_fs = int(eeg_raw.info['sfreq'])

    start_time, stop_time = time_range
    eeg_raw = eeg_raw.copy().crop(tmin=start_time, tmax=stop_time)
    print(f"EEG data cropped to {stop_time - start_time:.1f} seconds.")

    best_channel = None
    best_result = None
    best_smoothed = None

    # filter channel_list to only include channels that are in the EEG data
    available_channels = eeg_raw.ch_names
    # check if some of the available channels are in channel_list, if not, let user select a channel
    channel_list = [ch for ch in channel_list if ch in available_channels]
    if len(channel_list) == 0:
        print("⚠️ No valid channels found in the EEG data.")
        print(f"Available channels: {available_channels}")
        channel = input("Please select a channel: ")
        if channel not in available_channels:
            print("Invalid channel. Please select a valid channel.")
            return None, [], [], [], None
    
    for ch in channel_list:
        try:
            eeg_power, _ = compute_samplewise_eeg_power(
                eeg_raw,
                freq_low=freq_low,
                freq_high=freq_high,
                channel=ch
            )
        except Exception as e:
            print(f"⚠️ Skipping invalid channel: {ch}: {e}")
            continue

        smoothed = savgol_filter(eeg_power, window_length=smooth_window, polyorder=3)
        window_size = int(window_size_sec * eeg_fs)
    
        diffs = []
        for i in range(0, len(smoothed) - 2 * window_size):
            pre = smoothed[i : i + window_size]
            post = smoothed[i + window_size : i + 2 * window_size]
            diff = np.mean(post) - np.mean(pre)
            diffs.append((i + window_size, diff))

        if not diffs:
            continue

        idx, val = max(diffs, key=lambda x: abs(x[1]))
        result = {
            "type": "drop" if val < 0 else "spike",
            "index": idx,
            "time": start_time + idx / eeg_fs,
            "magnitude": val,
        }

        if best_result is None or abs(result["magnitude"]) > abs(best_result["magnitude"]):
            best_channel = ch
            best_result = result
            best_smoothed = smoothed

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(best_smoothed, label="Smoothed Power")
        plt.axvline(best_result["index"], color="red" if best_result["type"] == "drop" else "green",
                    linestyle="--", label=f"{best_result['type']} at {best_result['time']:.2f}s")
        plt.title(f"EEG Sync Detection - {best_channel}")
        plt.xlabel("Samples")
        plt.ylabel("Power")
        plt.legend()
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            dat = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(save_dir, f"{dat}_sync_onset_{best_channel}.png"))
            np.savetxt(f"outputs/outputData/{dat}_eeg_power.csv",
                       np.column_stack((np.arange(len(best_smoothed)), best_smoothed)), delimiter=",")
        print("---\nPlease close the EEG sync plot to continue.\n---")

        plt.show()

    return (
        best_channel,
        best_result["index"],
        best_result["time"],
        best_result,
        best_smoothed
    )


def confirm_sync_selection(
    channel: str, 
    sync_time: float, 
    result_type: str, 
    magnitude: float
) -> bool:
    """
    Ask the user to confirm the detected sync point.

    Args:
        channel (str): The selected EEG channel.
        sync_time (float): Time of the detected sync point (in seconds).
        result_type (str): 'drop' or 'spike'.
        magnitude (float): Magnitude of the change.

    Returns:
        bool: True if confirmed by user, False if manual selection is preferred.
    """

    print("\n🔍 Sync Artifact Detected")
    print("----------------------------------")
    print(f"  Channel   : {channel}")
    print(f"  Time      : {sync_time:.2f} s")
    print(f"  Type      : {result_type}")
    print(f"  Magnitude : {magnitude:.4f}")
    
    while True:
        user_input = input("\n✅ Use this EEG sync point? (yes/no): ").strip().lower()
        if user_input in ['y', 'yes']:
            return True
        elif user_input in ['n', 'no']:
            print("🛠️ Switching to manual sync selection...")
            return False
        else:
            print("Please enter 'yes' or 'no'.")
            