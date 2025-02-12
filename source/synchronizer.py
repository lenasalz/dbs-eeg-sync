# synchronizer.py
# Functions for synchronizing EEG and DBS data.

import numpy as np
import pandas as pd
import mne
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample

def synchronize_eeg_dbs(eeg_data, dbs_data, eeg_fs, dbs_fs, peak_index_eeg_fs):
    """
    Synchronizes EEG and DBS data by cropping both at the detected peak and resampling DBS to match EEG length.

    Args:
        eeg_data (mne.io.Raw): The EEG data object.
        dbs_data (pd.DataFrame): The DBS time series data.
        eeg_fs (float): Sampling frequency of the EEG.
        dbs_fs (float): Sampling frequency of the DBS.
        peak_index_eeg_fs (int): Peak index in EEG data.

    Returns:
        mne.io.Raw: Cropped EEG data starting from the peak.
        pd.DataFrame: Cropped and resampled DBS data aligned with EEG.
    """

    # Convert peak index in EEG samples to time in seconds
    peak_time_sec = peak_index_eeg_fs / eeg_fs

    # Convert peak time to DBS sample index
    peak_index_dbs_fs = int(peak_time_sec * dbs_fs)

    # Crop EEG from detected peak onwards
    cropped_eeg = eeg_data.copy().crop(tmin=peak_time_sec)

    # Crop DBS from detected peak onwards
    cropped_dbs = dbs_data.iloc[peak_index_dbs_fs:].reset_index(drop=True)

    # Get new EEG length after cropping
    num_samples_eeg = len(cropped_eeg.times)

    # Resample cropped DBS to match new EEG length
    resampled_dbs_values = resample(cropped_dbs["TimeDomainData"].values, num_samples_eeg)

    # Create a new index for the resampled DBS to match the length
    cropped_dbs = cropped_dbs.iloc[:num_samples_eeg].copy()  # Trim extra rows if necessary
    cropped_dbs["TimeDomainData"] = resampled_dbs_values

    print(f"EEG and DBS data cropped at {peak_time_sec} sec and resampled to match lengths.")

    return cropped_eeg, cropped_dbs


def resample_eeg_dbs(eeg_data, dbs_data, eeg_fs, dbs_fs):
    """
    Resamples EEG and DBS signals to match each other.

    Args:
        eeg_data (mne.io.Raw | np.array): The EEG signal or data array.
        dbs_data (pd.DataFrame | np.array): The DBS signal or data array.
        eeg_fs (float): Sampling frequency of the EEG.
        dbs_fs (float): Sampling frequency of the DBS.

    Returns:
        np.array, np.array: Resampled EEG and DBS signals.
    """

    eeg_samples = len(eeg_data.get_data()[0]) if isinstance(eeg_data, mne.io.Raw) else len(eeg_data)
    dbs_samples = len(dbs_data)

    if eeg_fs > dbs_fs:
        resampled_eeg = resample(eeg_data.get_data()[0], dbs_samples) if isinstance(eeg_data, mne.io.Raw) else resample(eeg_data, dbs_samples)
        return resampled_eeg, dbs_data
    elif dbs_fs > eeg_fs:
        resampled_dbs = resample(dbs_data, eeg_samples)
        return eeg_data, resampled_dbs
    else:
        return eeg_data, dbs_data
    

def find_eeg_peak(raw_eeg, freq_low, freq_high, decim, duration_sec=120, save_dir=None, log_file="sync_log.txt"):
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

    # Crop and filter EEG
    raw_cropped = raw_eeg.copy().crop(tmax=duration_sec)  
    raw_filtered = raw_cropped.copy().filter(l_freq=freq_low, h_freq=freq_high)  

    fs = raw_filtered.info["sfreq"]
    freqs = np.arange(freq_low, freq_high, 1)
    n_cycles = freqs / 1.5  
    time_bandwidth = 4.0  

    power = mne.time_frequency.tfr_multitaper(
        raw_filtered,
        picks="eeg",
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        decim=decim,
        average=True,
        return_itc=False
    )

    power_band_sum = power.data.mean(axis=0).sum(axis=0)  
    time_axis = np.arange(len(power_band_sum)) * decim / fs

    # Detect peaks
    peaks, peak_properties = find_peaks(power_band_sum, height=np.mean(power_band_sum) * 1.5)

    if len(peaks) > 0:
        # Select the peak with the highest power value
        highest_peak_idx = np.argmax(peak_properties["peak_heights"])
        peak_power_idx = peaks[highest_peak_idx]
    else:
        # If no peak is found, use the maximum value in the first 1000 samples as a fallback
        peak_power_idx = np.argmax(power_band_sum[:1000])  

    peak_index_eeg_fs = peak_power_idx * decim
    peak_index_eeg_s = peak_index_eeg_fs / fs

    # Log the detected peak
    with open(log_file, "a") as log:
        log.write(f"Detected highest EEG peak at {peak_index_eeg_fs} samples ({peak_index_eeg_s:.2f} sec)\n")

    print(f"Peak time logged in {log_file}")

    # Plot the detected peak
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, power_band_sum, label="Power in Band (120-130 Hz)")
    plt.axvline(time_axis[peak_power_idx], color='r', linestyle='--', label=f'Highest Peak @ {time_axis[peak_power_idx]:.2f} sec')
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.title('EEG: Power Over Time in 125 Hz Band')
    plt.legend()

    if save_dir:
        plt.savefig(f"{save_dir}/syncPSD.png")
        print(f"Plot saved to {save_dir}/syncPSD.png")

    plt.show()  # Ensures the plot opens

    return peak_index_eeg_fs, peak_index_eeg_s


def find_dbs_peak(dbs_data, dbs_fs, save_dir=None, log_file="sync_log.txt"):
    """
    Finds the highest synchronization peak index in DBS data.

    Args:
        dbs_data (pd.DataFrame): The DBS time series data.
        dbs_fs (float): Sampling frequency of the DBS.
        save_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
        log_file (str, optional): Path to the log file where peak detection results will be saved.

    Returns:
        int: Peak index in DBS sampling frequency.
        float: Peak time in seconds.
        pd.DataFrame: Cropped DBS data from the detected peak onward.
    """

    # Extract DBS signal
    dbs_signal = dbs_data["TimeDomainData"].values

    # Time axis
    dbs_time_axis = np.arange(len(dbs_signal)) / dbs_fs

    # Find peaks in DBS signal
    peaks, peak_properties = find_peaks(np.abs(dbs_signal), height=np.mean(np.abs(dbs_signal)) * 1.5)

    if len(peaks) > 0:
        # Select the peak with the highest amplitude
        highest_peak_idx = np.argmax(peak_properties["peak_heights"])
        peak_dbs_idx = peaks[highest_peak_idx]
    else:
        # If no peak is found, use the maximum amplitude location as a fallback
        peak_dbs_idx = np.argmax(np.abs(dbs_signal[:1000]))

    peak_dbs_time = peak_dbs_idx / dbs_fs

    # Log the detected peak
    with open(log_file, "a") as log:
        log.write(f"Detected highest DBS peak at {peak_dbs_idx} samples ({peak_dbs_time:.2f} sec)\n")

    print(f"DBS peak time logged in {log_file}")

    # Plot the detected DBS peak
    plt.figure(figsize=(10, 5))
    plt.plot(dbs_time_axis, dbs_signal, label="DBS Signal")
    plt.axvline(dbs_time_axis[peak_dbs_idx], color='r', linestyle='--', label=f'Highest Peak @ {dbs_time_axis[peak_dbs_idx]:.2f} sec')
    plt.xlabel('Time (s)')
    plt.ylabel('DBS Amplitude')
    plt.title('DBS Peak Detection')
    plt.legend()

    if save_dir:
        plt.savefig(f"{save_dir}/dbs_peak.png")
        print(f"Plot saved to {save_dir}/dbs_peak.png")

    plt.show()  # Ensures the plot opens

    # Crop DBS signal at detected peak
    cropped_dbs = dbs_data.iloc[peak_dbs_idx:].reset_index(drop=True)

    return peak_dbs_idx, peak_dbs_time, cropped_dbs


def plot_synchronized_signals(eeg_data, dbs_data, peak_time_sec, save_dir=None):
    """
    Plots the synchronized EEG and DBS signals overlayed.

    Args:
        eeg_data (mne.io.Raw): Synchronized EEG data.
        dbs_data (pd.DataFrame): Synchronized DBS data.
        peak_time_sec (float): Time at which peak was detected.
        save_dir (str, optional): Directory to save the plot.
    """

    # Get EEG and DBS time vectors
    eeg_times = eeg_data.times - peak_time_sec  # Align EEG to peak
    dbs_times = np.linspace(0, len(dbs_data) / 250, len(dbs_data))  # Assuming DBS was at 250Hz

    # Get EEG signal (first channel for visualization)
    eeg_signal = eeg_data.get_data()[0]

    # Get DBS signal
    dbs_signal = dbs_data["TimeDomainData"].values

    # Plot both signals
    plt.figure(figsize=(12, 5))
    plt.plot(eeg_times, eeg_signal, label="EEG Signal", color='blue', alpha=0.7)
    plt.plot(dbs_times, dbs_signal, label="DBS Signal", color='orange', alpha=0.7)
    plt.axvline(0, color='r', linestyle='--', label='Detected Peak')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.title('Synchronized EEG & DBS Signals')
    plt.legend()

    if save_dir:
        plt.savefig(f"{save_dir}/eeg_dbs_overlay.png")
        print(f"Overlay plot saved to {save_dir}/eeg_dbs_overlay.png")

    plt.show()