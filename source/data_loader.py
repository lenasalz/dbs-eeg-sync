import os
import mne  # Assuming EEG data uses MNE library
import json
import pandas as pd
import numpy as np
import pyxdf
from typing import Tuple

def load_eeg_data(file_path: str):
    """
    Loads EEG data from various file formats using MNE.

    Args:
        file_path (str): Path to the EEG file.

    Returns:
        mne.io.Raw | mne.Epochs | mne.Evoked: The loaded EEG data object.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".set":
        raw = mne.io.read_raw_eeglab(file_path, preload=True)  
    elif ext == ".fif":
        raw = mne.io.read_raw_fif(file_path, preload=True)
    elif ext in [".vhdr", ".eeg", ".vmrk"]:
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(file_path, preload=True)
    elif ext == ".gdf":
        raw = mne.io.read_raw_gdf(file_path, preload=True)
    elif ext == ".cnt":
        raw = mne.io.read_raw_cnt(file_path, preload=True)
    elif ext == ".mff":
        raw = mne.io.read_raw_egi(file_path, preload=True)
    elif ext == ".xdf":
        streams, _ = pyxdf.load_xdf(file_path)
        eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
        if eeg_stream is None:
            raise ValueError("No EEG stream found in the .xdf file")
        data = np.array(eeg_stream['time_series']).T
        ch_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']]
        sfreq = float(eeg_stream['info']['nominal_srate'][0])
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    print(f"---\nSuccessfully loaded {file_path}")
    return raw


def dbs_artifact_settings():
    """ 
    Asks user to input the frequency range and duration for the DBS artifact detection.
    If no input is given, the default values are used.
    Default values: dbs_freq_min = 120, dbs_freq_max = 130, duration_sec = 120
    """
    # ask user if he wants to adapt the filter frequency range, otherwise use default values are taken
    if input("---\nThe default filter frequencies for frequency DBS artifact detection are 120 - 130 Hz, in the first 120 seconds. \nDo you want to adapt these? (yes/no): ").strip().lower() == "yes":
        dbs_freq_min = int(input("---\nEnter the minimum frequency for DBS artifact detection (usually 120): ").strip())
        dbs_freq_max = int(input("---\nEnter the maximum frequency for DBS artifact detection (usually 130): ").strip())
        dbs_duration_sec = int(input("---\nEnter the duration of the DBS signal for artifact detection (in seconds): ").strip())
    else:
        dbs_freq_min = 120
        dbs_freq_max = 130
        dbs_duration_sec = 120

    return dbs_freq_min, dbs_freq_max, dbs_duration_sec


def open_json_file(filepath: str) -> dict:
    """
    Opens a JSON file and returns its contents as a dictionary.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Dictionary containing the JSON data.
    """
    with open(filepath) as jsonfile:
        data = json.load(jsonfile)
    return data


def select_recording(json_data: dict) -> int:
    """
    Prompts the user to select a recording from the JSON data.

    Args:
        json_data (dict): Parsed JSON data.

    Returns:
        int: Selected recording number.
    
    Raises:
        ValueError: If no recordings are available for selection.
    """
    recordings = json_data.get("BrainSenseTimeDomain", [])
    n_recordings = len(recordings)

    if n_recordings == 0:
        raise ValueError("No recordings available for selection.")

    print(f"---\nAvailable recordings: {list(range(n_recordings))}")

    while True:
        try:
            rec_num = input(f"Enter the recording number to read (0-{n_recordings-1}): ").strip()
            
            if not rec_num:
                print("---\nInput cannot be empty. Please enter a valid recording number.")
                continue
            
            rec_num = int(rec_num)
            print(f"---\nReading recording {rec_num}...")

            if 0 <= rec_num < n_recordings:
                return rec_num
            else:
                print("---\nInvalid recording number. Please try again.")
        except ValueError:
            print("---\nInvalid input. Please enter a valid recording number.")


def read_time_domain_data(json_data: dict, rec_num: int) -> tuple:
    """
    Extracts and returns TimeDomainData from the selected recording.

    Args:
        json_data (dict): Parsed JSON data.
        rec_num (int): Selected recording number.

    Returns:
        tuple: (DataFrame containing the time domain data, sampling frequency in Hz).
    """
    fs = json_data["BrainSenseTimeDomain"][rec_num].get("SampleRateInHz")
    
    df = pd.DataFrame(json_data["BrainSenseTimeDomain"][rec_num]["TimeDomainData"], columns=["TimeDomainData"])
    df["recording"] = rec_num
    df["SampleRateInHz"] = fs
    print(f"---\nSuccessfully read DBS recording {rec_num} with sampling frequency {fs} Hz")
    return df


def read_lfp_data(json_data: dict, rec_num: int) -> tuple:
    """
    Extracts and returns LfpData from the selected block.

    Args:
        json_data (dict): Parsed JSON data.
        rec_num (int): Selected recording number.

    Returns:
        tuple: (DataFrame containing the LFP data, sampling frequency in Hz, lead side).
    """
    fs = json_data["BrainSenseLfp"][rec_num].get("SampleRateInHz")
    channel = json_data["BrainSenseLfp"][rec_num]["Channel"]
    lead = "Left" if "LEFT" in channel else "Right" if "RIGHT" in channel else "Unknown"
    
    lfp_data = [sample[lead].get("LFP") for sample in json_data["BrainSenseLfp"][rec_num]["LfpData"]]
    df = pd.DataFrame(lfp_data, columns=["LfpData"])
    df["recording"] = rec_num
    print(f"---\nSuccessfully read DBS recording {rec_num} with sampling frequency {fs} Hz")
    return df, fs, lead



def compute_eeg_power(
        raw_eeg: mne.io.Raw, 
        freq_low: int, 
        freq_high: int, 
        duration_sec: int=120, 
        channel: str="Cz"
        ) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Computes the power of EEG data in a specified frequency range and duration.

    Args:
        raw_eeg (mne.io.Raw): The raw EEG data.
        freq_low (int): Lower frequency bound for power computation.
        freq_high (int): Upper frequency bound for power computation.
        duration_sec (int, optional): Duration in seconds for which to compute the power. Defauls to 120 seconds.
        channel (str, optional): The EEG channel to analyze. Defaults to "Cz".

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
            - eeg_power_band_sum: The computed power in the specified frequency range.
            - power_time_axis: Time axis corresponding to the computed power.
    """

    # Crop and filter EEG
    _raw = raw_eeg.copy()
    raw_cropped = _raw.crop(tmax=duration_sec)
    raw_filtered = raw_cropped.filter(l_freq=freq_low, h_freq=freq_high)

    fs = raw_filtered.info["sfreq"]
    # check nyqist and adjust frequency range
    nyquist_freq = fs / 2
    if freq_low > nyquist_freq or freq_high > nyquist_freq:
        raise ValueError(f"Frequency range exceeds Nyquist frequency: {nyquist_freq} Hz")
    if freq_low < 0 or freq_high < 0:
        raise ValueError("Frequency range must be positive")
    if freq_low >= freq_high:
        raise ValueError("Frequency low must be less than frequency high")
    # Define frequencies and wavelet parameters
    freqs = np.arange(freq_low, freq_high, 1)
    n_cycles = freqs / 1.5  
    time_bandwidth = 4.0  
    picks = channel    # only one channel is used for the analysis

    power = mne.time_frequency.tfr_multitaper(
        raw_filtered,
        # picks="eeg",
        picks=picks,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        decim=1,
        average=True,
        return_itc=False
    )

    eeg_power_band_sum = power.data.mean(axis=0).sum(axis=0)  

    # for the peak detection, ignore the first and last second, as there are artefacts from power computation
    cut_samples = int((fs) * 1)
    eeg_power_band_sum = eeg_power_band_sum[cut_samples:-cut_samples]

    power_time_axis = np.arange(len(eeg_power_band_sum)) / fs

    return eeg_power_band_sum, power_time_axis


if __name__ == "__main__":
    file_path = "/Users/lenasalzmann/dev/dbs-eeg-sync/data/eeg_example.set"
    eeg_data = load_eeg_data(file_path)
    print(eeg_data)
