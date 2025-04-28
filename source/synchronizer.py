# synchronizer.py
# Functions for synchronizing EEG and DBS data.

import numpy as np
import pandas as pd
import mne
import os
import matplotlib.pyplot as plt
from scipy.signal import resample
from datetime import datetime



def crop_data(eeg_data, dbs_data, peak_dbs_idx, peak_index_eeg_fs):
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


def synchronize_data(cropped_eeg, cropped_dbs, save_dir=None):
    """
    Synchronizes EEG and DBS data by resampling to the same length efficiently.

    Args:
        cropped_eeg (mne.io.Raw): EEG data (MNE object).
        cropped_dbs (pd.DataFrame): DBS data (Pandas DataFrame).
        save_dir (str, optional): Directory to save the plot.

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

    # Generate time vectors efficiently
    n_times = resampled_eeg.get_data().shape[1]
    eeg_times = np.linspace(0, n_times / target_fs, n_times)
    dbs_times = np.linspace(0, len(resampled_dbs_signal) / target_fs, len(resampled_dbs_signal))

    # Plot the signals
    plt.figure(figsize=(12, 5))
    plt.plot(eeg_times, resampled_eeg.get_data()[0], label="EEG Signal (Channel 0)", color='blue', alpha=0.7)
    plt.plot(dbs_times, resampled_dbs_signal, label="DBS Signal", color='orange', alpha=0.7)
    plt.axvline(0, color='r', linestyle='--', label='Detected Peak')

    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.title('Synchronized EEG & DBS Signals')
    plt.legend()

    if save_dir:
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/eeg_dbs_overlay_{dat}.png")
        print(f"---\nOverlay plot saved to {save_dir}/eeg_dbs_overlay_{dat}.png")
    
    print("---\nPlease close the plot to continue.")

    plt.show()

    # Update DBS data
    synchronized_dbs = cropped_dbs.copy()
    synchronized_dbs["TimeDomainData"] = resampled_dbs_signal
    synchronized_dbs["SampleRateInHz"] = target_fs

    return resampled_eeg, synchronized_dbs


def save_synchronized_data(synchonized_eeg, synchronized_dbs, output_dir="outputs/outputData"):
    """
    Saves the synchronized EEG and DBS data.

    Args:
        eeg_data (mne.io.Raw): Synchronized EEG data.
        dbs_data (pd.DataFrame): Synchronized DBS data.
        output_dir (str, optional): Directory to save the files. Defaults to "data".

    Returns:
        None
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not isinstance(synchonized_eeg, mne.io.BaseRaw):
        raise ValueError("eeg_data is not an instance of mne.io.Raw")

    # Save EEG data in .fif format (MNE format)
    eeg_output_path = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + "_synchronized_eeg.fif")
    synchonized_eeg.save(eeg_output_path, overwrite=True)
    print(f"---\nSaved synchronized EEG to {eeg_output_path}")

    # Save DBS data as CSV
    dbs_output_path = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + "_synchronized_dbs.csv")
    synchronized_dbs.to_csv(dbs_output_path, index=False)
    print(f"---\nSaved synchronized DBS to {dbs_output_path}")

