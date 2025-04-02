# sync_peak_finder.py
# Functions for finding synchronization peaks in EEG and DBS data.
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import csv

def find_eeg_peak(eeg_power_band_sum, fs, power_time_axis, save_dir=None, log_file="sync_log.txt"):

    """
    Finds the highest synchronization peak index in EEG data within a specified frequency band.

    Args:
        raw_eeg (mne.io.Raw): The raw EEG data object to analyze.
        freq_low (float): Lower bound of the frequency range of interest.
        freq_high (float): Upper bound of the frequency range of interest.
        decim (int): Decimation factor for downsampling.
        duration_sec (int, optional): Duration of the cropped EEG window (default: 120 sec).
        save_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
        log_file (str, optional): Path to the log file where peak detection results will be saved.

    Returns:
        int: Peak index in EEG sampling frequency.
        float: Peak time in seconds.
    """

    # Detect peaks
    peaks, peak_properties = find_peaks(eeg_power_band_sum, height=np.mean(eeg_power_band_sum) * 1.5)

    if len(peaks) > 0:
        # Select the peak with the highest power value
        highest_peak_idx = np.argmax(peak_properties["peak_heights"])
        peak_power_idx = peaks[highest_peak_idx]
    else:
        # If no peak is found, use the maximum value in the first 1000 samples as a fallback
        peak_power_idx = np.argmax(eeg_power_band_sum[:1000])  

    eeg_peak_index_fs = int(peak_power_idx)
    eeg_peak_index_s = eeg_peak_index_fs / fs

    # Log the detected peak
    with open(log_file, "a") as log:
        log.write(f"Detected highest EEG peak at {eeg_peak_index_fs} samples ({eeg_peak_index_s:.2f} sec)\n")

    print(f"---\nPeak time logged in {log_file}")

    # Plot the detected peak
    plt.figure(figsize=(10, 5))
    plt.plot(power_time_axis, eeg_power_band_sum, label="EEG Power")
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


def find_dbs_peak(dbs_signal, dbs_fs, save_dir=None, log_file="sync_log.txt"):
    """
    Finds the highest peak in DBS data in the positive direction only.

    Args:
        dbs_signal (): The DBS time series data.
        dbs_fs (float): Sampling frequency of the DBS.
        save_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
        log_file (str, optional): Log file path for saving detected peak info.

    Returns:
        int: Peak index in DBS samples.
        float: Peak time in seconds.
        pd.DataFrame: Cropped DBS data from the detected peak onward.
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

    # Log detected peak
    with open(log_file, "a") as log:
        log.write(f"Detected DBS peak at {dbs_peak_index_fs} samples ({dbs_peak_index_s:.2f} sec)\n")

    print(f"---\nDBS peak time logged in {log_file}")

    # Plot detected peak
    plt.figure(figsize=(10, 5))
    plt.plot(dbs_time_axis, dbs_signal, label="DBS Signal")
    plt.axvline(dbs_time_axis[dbs_peak_index_fs], color='r', linestyle='--', label=f'Peak @ {dbs_peak_index_s:.2f} sec')
    plt.xlabel('Time (s)')
    plt.ylabel('DBS Amplitude')
    plt.title('DBS Peak Detection')
    plt.legend()

    if save_dir:
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/syncPeakDBS_{dat}.png")
        print(f"---\nPlot saved to {save_dir}/syncPeakDBS_{dat}.png")
    
    plt.show()

    return dbs_peak_index_fs, dbs_peak_index_s


def save_sync_peak_info(eeg_file, dbs_file, eeg_peak_idx, eeg_peak_time, dbs_peak_idx, dbs_peak_time, output_file="sync_info.csv"):
    """
    Save EEG & DBS peak indices and times with file names to a CSV table.
    """
    header = ["EEG File", "DBS File", "EEG Peak Index", "EEG Peak Time (s)", "DBS Peak Index", "DBS Peak Time (s)"]
    row = [eeg_file, dbs_file, eeg_peak_idx, eeg_peak_time, dbs_peak_idx, dbs_peak_time]

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{output_file}"
    
    try:
        write_header = not os.path.isfile(output_path)

        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
        print(f"---\nPeak info saved to {output_path}")

    except Exception as e:
        print(f"Error saving peak info: {e}")


