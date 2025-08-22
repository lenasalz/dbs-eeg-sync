# synchronizer.py
# Functions for synchronizing EEG and DBS data.

import numpy as np
import pandas as pd
import mne
import os
import csv
import matplotlib.pyplot as plt
from scipy.signal import resample
from datetime import datetime


def cut_data_at_sync(eeg_data, dbs_data, peak_dbs_idx, peak_index_eeg_fs):
    # ToDo: make it work with numpy arrays
    """ 
    Crops the data at the detected peaks and resamples to match the shortest signal.

    Args:
        eeg_data (mne.io.Raw): The EEG data object.
        dbs_data (pd.DataFrame): The DBS time series data.
        peak_dbs_idx (int): Peak index in DBS samples.
        peak_index_eeg_fs (int): Peak index in EEG samples.

    Returns:
        mne.io.Raw: Cropped and resampled EEG data.
        pd.DataFrame: Cropped and resampled DBS data.    
    
    """

    cropped_eeg = eeg_data.copy().crop(tmin=peak_index_eeg_fs / eeg_data.info["sfreq"])
    cropped_dbs = dbs_data.iloc[peak_dbs_idx:].reset_index(drop=True)

    eeg_fs = eeg_data.info["sfreq"]
    dbs_fs = dbs_data["SampleRateInHz"][0]

    # check length
    print(f"---\nCropped EEG length: {len(cropped_eeg.times)/eeg_fs/60} minutes")
    print(f"---\nCropped DBS length: {len(cropped_dbs)/dbs_fs/60} minutes")

    return cropped_eeg, cropped_dbs



def synchronize_data(cropped_eeg, cropped_dbs, resample_data=True, save_dir="outputs/plots", sub_id=None, block=None):
    
    """
    Synchronizes EEG and DBS data by resampling to the same length efficiently.

    Args:
        cropped_eeg (mne.io.Raw): EEG data (MNE object).
        cropped_dbs (pd.DataFrame): DBS data (Pandas DataFrame).
        resample_data (bool, optional): Whether to resample the data. Defaults to True. If not given, it will ask the user if they want to resample the data.
        save_dir (str, optional): Directory to save the plot. Defaults to "outputs/plots".
        sub_id (str): The subject ID.
        block (str): The block name.

    Returns:
        synchronized_eeg (mne.io.Raw): Resampled EEG data with updated sampling frequency.
        synchronized_dbs (pd.DataFrame): Resampled DBS data.
    """

    # --- Assertions ---
    assert isinstance(cropped_eeg, mne.io.BaseRaw), "cropped_eeg is not an instance of mne.io.Raw"
    assert isinstance(cropped_dbs, pd.DataFrame), "cropped_dbs is not a Pandas DataFrame"
    assert cropped_dbs.shape[0] > 0, "cropped_dbs is empty"
    assert cropped_eeg.get_data().shape[1] > 0, "cropped_eeg is empty"
    if save_dir is not None:
        assert isinstance(save_dir, str), "save_dir must be a string"
        assert os.path.isdir(save_dir), f"save_dir '{save_dir}' does not exist or is not a directory"

    # --- Synchronize EEG and DBS data ---
    # Get EEG sampling rate
    eeg_fs = cropped_eeg.info["sfreq"]

    # Get DBS signal & sampling rate
    dbs_signal = cropped_dbs["TimeDomainData"].values
    dbs_fs = cropped_dbs["SampleRateInHz"].iloc[0]

    # Choose the target sampling frequency (lower one for efficiency)
    target_fs = min(eeg_fs, dbs_fs)

    # Ask user if they want to resample the data to the target sampling frequency
    if resample_data is None:

        resample_data = input("---\nResample data to the lower sampling frequency? (yes/no): ").strip().lower()
    if resample_data == "yes":
        # Resample EEG using MNE's built-in method if needed
        if eeg_fs != target_fs:
            resampled_eeg = cropped_eeg.resample(sfreq=target_fs)
        else:
            resampled_eeg = cropped_eeg
        # Resample DBS signal if needed
        if dbs_fs != target_fs:
            resampled_dbs_signal = resample(dbs_signal, int(len(dbs_signal) * target_fs / dbs_fs))
        else:
            resampled_dbs_signal = dbs_signal
    else:
        resampled_eeg = cropped_eeg
        resampled_dbs_signal = dbs_signal

    # Generate time vectors efficiently
    n_times = resampled_eeg.get_data().shape[1]

    if resample_data == "yes":
        eeg_times = np.linspace(0, n_times / target_fs, n_times)
        dbs_times = np.linspace(0, len(resampled_dbs_signal) / target_fs, len(resampled_dbs_signal))
    else:
        eeg_times = np.arange(n_times) / eeg_fs
        dbs_times = np.arange(len(resampled_dbs_signal)) / dbs_fs

    # Plot the signals as overlay
    plt.figure(figsize=(12, 5))
    plt.plot(eeg_times, resampled_eeg.get_data()[0], label="EEG Signal (Channel 0)", color='orange', alpha=0.7)
    plt.plot(dbs_times, resampled_dbs_signal, label="DBS Signal", color='blue', alpha=0.7)
    plt.axvline(0, color='r', linestyle='--', label='Detected Peak')

    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.title(f'Synchronized EEG & DBS Signals - {sub_id} | {block}')
    plt.legend()

    if save_dir:
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/{dat}_eeg_dbs_overlay_{sub_id}_{block}.png")
        print(f"---\nOverlay plot saved to {save_dir}/eeg_dbs_overlay_{sub_id}_{block}_{dat}.png")
    
    print("---\nPlease close the overlay plot to continue.")

    plt.show()

    # Update DBS data
    synchronized_dbs = cropped_dbs.copy()
    synchronized_dbs["TimeDomainData"] = resampled_dbs_signal
    synchronized_dbs["SampleRateInHz"] = target_fs

    return resampled_eeg, synchronized_dbs


def save_synchronized_data(synchonized_eeg, synchronized_dbs, output_dir="outputs/outputData", sub_id=None, block=None):
    """
    Saves the synchronized EEG and DBS data.

    Args:
        eeg_data (mne.io.Raw): Synchronized EEG data.
        dbs_data (pd.DataFrame): Synchronized DBS data.
        output_dir (str, optional): Directory to save the files. Defaults to "outputs/outputData".
        sub_id (str): The subject ID.
        block (str): The block name.

    Returns:
        None
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not isinstance(synchonized_eeg, mne.io.BaseRaw):
        raise ValueError("eeg_data is not an instance of mne.io.Raw")

    # Save EEG data in .fif format (MNE format)

    eeg_output_path = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{sub_id}_{block}_synchronized_eeg.fif")
    synchonized_eeg.save(eeg_output_path, overwrite=True)
    print(f"---\nSaved synchronized EEG to {eeg_output_path}")

    # Save DBS data as CSV
    dbs_output_path = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{sub_id}_{block}_synchronized_dbs.csv")
    synchronized_dbs.to_csv(dbs_output_path, index=False)
    print(f"---\nSaved synchronized DBS to {dbs_output_path}")


def save_sync_info(
    sub_id,
    block,
    eeg_file=None,
    dbs_file=None,
    eeg_peak_idx=None,
    eeg_peak_time=None,
    dbs_peak_idx=None,
    dbs_peak_time=None,
    output_file="sync_info.csv"
):
    """
    Save synchronization info to a CSV.

    Args:
        sub_id (str): Subject ID.
        block (str): Block name.
        eeg_file (str): EEG filename.
        dbs_file (str): DBS filename.
        eeg_peak_idx (int): EEG peak index (or manual selection).
        eeg_peak_time (float): EEG peak time in seconds.
        dbs_peak_idx (int): DBS peak index.
        dbs_peak_time (float): DBS peak time in seconds.
        output_file (str): Output CSV file name.
    """
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    timestamp = datetime.now().isoformat()

    header = [
        "timestamp", "sub_id", "block",
        "EEG File", "DBS File",
        "EEG Peak Index", "EEG Peak Time (s)",
        "DBS Peak Index", "DBS Peak Time (s)"
    ]

    row = [
        timestamp,
        sub_id,
        block,
        eeg_file,
        dbs_file,
        eeg_peak_idx,
        eeg_peak_time,
        dbs_peak_idx,
        dbs_peak_time
    ]

    write_header = True
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            first_line = f.readline()
            if first_line.strip():
                write_header = False

    try:
        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
        print(f"---\nSync info saved to {output_path}")
    except Exception as e:
        print(f"Error saving sync info: {e}")


if __name__ == "__main__":
    # load eeg_power.csv
    eeg_power = pd.read_csv("outputs/outputData/eeg_power.csv", index_col=0)
    # load dbs_signal.csv
    dbs_signal = pd.read_csv("outputs/outputData/dbs_signal.csv", index_col=0)
    # synchronize data
    synchronize_data(eeg_power, dbs_signal)