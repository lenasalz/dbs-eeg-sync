# sync_peak_finder.py
# Functions for finding synchronization peaks in EEG and DBS data.
import matplotlib.pyplot as plt
import numpy as np
import os
import mne
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import csv
from datetime import datetime
import plotly.graph_objects as go
from typing import Tuple, Optional

from source.power_calculater import compute_samplewise_eeg_power


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


def detect_eeg_change_window(
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


def select_eeg_channel(eeg_raw, freq_low, freq_high, channel="Cz"):
    """
    Selects a specific EEG channel and computes the power in a specified frequency band.

    Args:
        eeg_raw (mne.io.Raw): The raw EEG data.
        freq_low (float): Lower frequency bound for power computation.
        freq_high (float): Upper frequency bound for power computation.
        channel (str, optional): Name of the EEG channel to analyze. Defaults to "Cz".
    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the computed power and time axis.

    """

    if channel is None:
        eeg_channel_names = eeg_raw.ch_names
    else:
        eeg_channel_names = [channel]

    # select channels of interest
    eeg_channel_names = ['CPz', 'Pz', 'Oz']
    n_channels = len(eeg_channel_names)
    n_cols = 4

    best_channel = None
    best_magnitude = 0
    best_result = None
    best_smoothed = None

    for i, ch in enumerate(eeg_channel_names):
            try:
                eeg_power, _ = compute_samplewise_eeg_power(
                    eeg_raw, freq_low, freq_high, channel=ch
                )
            except Exception as e:
                print(f"Error processing {ch}: {e}")
                continue

    # Detect drop/spike
            result, smoothed = detect_eeg_change_window(
                eeg_power,
                eeg_fs=int(eeg_raw.info['sfreq']),
                plot=False
            )
            if result and abs(result["magnitude"]) > best_magnitude:
                best_channel = ch
                best_magnitude = abs(result["magnitude"])
                best_result = result
                best_smoothed = smoothed

            if best_channel:
                print(f"    Type: {best_result['type']}, Magnitude: {best_result['magnitude']:.3f}, Time: {best_result['time']:.2f}s")

    if best_channel is None:
        print("⚠️ No suitable channel found.")
        return None, None

    return best_channel, best_result, best_smoothed


def detect_sync_from_eeg(
    eeg_raw: mne.io.Raw,
    freq_low: float,
    freq_high: float,
    duration_sec: int = 120,
    eeg_fs: Optional[int] = None,
    channel_list: list = ["CPz", "Pz", "Oz", "Fz", "Cz", "T7", "T8", "O1", "O2"],
    smooth_window: int = 301,
    window_size_sec: int = 2,
    plot: bool = False,
    save_dir: Optional[str] = None,
) -> Tuple[Optional[str], Optional[int], Optional[float], Optional[dict], Optional[np.ndarray]]:
    """
    EEG samplewise power computation, best channel selection, and change detection (drop or spike).
    
    Args:
        eeg_raw (mne.io.Raw): Raw EEG data.
        freq_low (float): Lower frequency bound.
        freq_high (float): Upper frequency bound.
        duration_sec (float): Duration in seconds of the eeg signal to analyze. Defaults to 120 seconds.
        eeg_fs (int, optional): Sampling frequency. If None, extracted from eeg_raw.
        channel_list (list): List of channels to consider. Defaults to ["CPz", "Pz", "Oz", "Fz", "Cz", "T7", "T8", "O1", "O2"]. If None, all channels are used.
        smooth_window (int): Window size for smoothing. Defaults to 301.
        window_size_sec (int): Duration in seconds for diff window. Defaults to 2.
        plot (bool): Whether to plot the smoothed power with sync /change idx marker.
        save_dir (str, optional): Directory to save the plot.
    
    Returns:
        Tuple containing:
            - Best channel name (str)
            - sync_idx (int): Index of detected drop/spike
            - sync_time (float): Time of detected drop/spike (seconds)
            - result (dict): Full result dict from detection
            - smoothed (np.ndarray): Smoothed power signal of best channel
    """

    if eeg_fs is None:
        eeg_fs = int(eeg_raw.info['sfreq'])

    # ToDo: add duration_sec as length of the eeg data

    # crop the eeg data to duration_sec, if given
    if duration_sec:
        start_time = 0
        stop_time = duration_sec
        eeg_raw.crop(tmin=start_time, tmax=stop_time)
        print(f"EEG data cropped to {duration_sec} seconds.")

    best_channel = None
    best_magnitude = 0
    best_result = None
    best_smoothed = None

    for ch in channel_list:
        try:
            eeg_power, _ = compute_samplewise_eeg_power(
                eeg_raw, freq_low, freq_high, channel=ch
            )
        except Exception as e:
            print(f"⚠️ Error processing {ch}: {e}")
            continue

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

        if result and abs(result["magnitude"]) > best_magnitude:
            best_channel = ch
            best_magnitude = abs(result["magnitude"])
            best_result = result
            best_smoothed = smoothed

    if best_channel is None:
        print("⚠️ No suitable channel found.")
        return None, None, None, None, None

    if plot:
        print("---\nPlease close the EEG sync plot to continue.\n---")
        plt.figure(figsize=(10, 4))
        plt.plot(best_smoothed, label="Smoothed Power")
        plt.axvline(best_result["index"], 
                    color="red" if best_result["type"] == "drop" else "green", 
                    linestyle="--", label=f"Sync Point, {best_result['time']:.2f}s, {best_result['index']} samples")
        plt.title(f"EEG Sync Point Detection ({best_result['type'].capitalize()}) - Channel: {best_channel}")
        plt.xlabel("Samples")
        plt.ylabel("Power")
        plt.legend()
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"change_onset_{best_channel}.png"))
            # save power to csv in outputs folder with an index column
            np.savetxt("outputs/outputData/eeg_power.csv", np.column_stack((np.arange(len(best_smoothed)), best_smoothed)), delimiter=",")
        plt.show()
        

    return best_channel, best_result["index"], best_result["time"], best_result, best_smoothed


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
            