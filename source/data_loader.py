import os
import mne  
import json
import pandas as pd
import numpy as np
import pyxdf


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
    file_path = file_path.strip('\'"')
    file_path = os.path.normpath(os.path.expanduser(file_path))
    
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
        # Get Marker/Event stream
        marker_stream = next((s for s in streams if s['info']['type'][0].lower() in ['marker', 'markers', 'trigger']), None)

        if marker_stream is not None:
            marker_times = np.array(marker_stream['time_stamps'])
            marker_labels = np.array(marker_stream['time_series']).flatten()

            # Convert marker timestamps to sample indices relative to EEG start
            eeg_start_time = eeg_stream['time_stamps'][0]
            marker_sample_indices = ((marker_times - eeg_start_time) * sfreq).astype(int)

            # Create event list for MNE
            event_id_dict = {label: idx + 1 for idx, label in enumerate(np.unique(marker_labels))}
            events = np.array([[idx, 0, event_id_dict[label]] for idx, label in zip(marker_sample_indices, marker_labels)])

            # Add annotations for visualization (optional)
            durations = [0] * len(marker_labels)
            descriptions = marker_labels.astype(str).tolist()
            annotations = mne.Annotations(onset=marker_sample_indices / sfreq, duration=durations, description=descriptions)
            raw.set_annotations(annotations)
        else:
            print("No marker stream found in the .xdf file")
            events = None
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    sfreq = raw.info['sfreq']
    print(f"---\nSuccessfully loaded {file_path}")
    return raw, sfreq


def dbs_artifact_settings():
    """ 
    Asks user to input the frequency range and time range for DBS artifact detection.
    Default: freq 120–130 Hz, time = full available range (None, None).
    """

# AFTER:
def dbs_artifact_settings(freq_min: float | None = None,
                          freq_max: float | None = None,
                          tmin: float | None = None,
                          tmax: float | None = None,
                          interactive: bool = False):
    """
    Frequency range and time range settings for DBS artifact detection.
    Default: freq 120–130 Hz, time = full available range (None, None).
    If interactive=False (default), return given values or sensible defaults
    without asking the user. If interactive=True, keep the current prompts.
    """

    if not interactive:
        fmin = 120.0 if freq_min is None else float(freq_min)
        fmax = 130.0 if freq_max is None else float(freq_max)
        return fmin, fmax, tmin, tmax

        
    answer = input(
        "---\nThe DBS default settings are: frequency 120–130 Hz, full time range.\n"
        "Do you want to adapt these? (yes/no): "
    ).strip().lower()

    if answer == "yes":
        # Frequency range
        dbs_freq_min = int(input("---\nEnter the minimum frequency (usually 120): ").strip())
        dbs_freq_max = int(input("---\nEnter the maximum frequency (usually 130): ").strip())

        # Time range
        time_input = input(
            "---\nEnter the DBS time range in seconds "
            "(e.g., '5-120', '120' for 0–120, or leave blank for full range): "
        ).strip()

        if time_input == "":
            start_time, end_time = None, None   # full range
        elif "-" in time_input:
            start_time, end_time = map(int, time_input.split("-"))
        else:
            start_time, end_time = 0, int(time_input)
    else:
        dbs_freq_min, dbs_freq_max = 120, 130
        start_time, end_time = None, None   # full range by default

    return dbs_freq_min, dbs_freq_max, start_time, end_time


def open_json_file(filepath: str) -> dict:
    """
    Opens a JSON file and returns its contents as a dictionary.

    Args:
        filepath: Path to the JSON file, accepts both string and Path objects.

    Returns:
        dict: Dictionary containing the JSON data.
    """
    if isinstance(filepath, os.PathLike):
        filepath = str(filepath)

    filepath = filepath.strip('\'"')
    filepath = os.path.normpath(os.path.expanduser(filepath))
    with open(filepath) as jsonfile:
        data = json.load(jsonfile)
    return data


def select_recording(json_data, index: int | None = None):
    """
    If index is provided, return it without prompting.
    Otherwise, fall back to interactive selection.
    """
    if index is not None:
        return int(index)

    recordings = json_data.get("BrainSenseTimeDomain", [])
    n_recordings = len(recordings)

    if n_recordings == 0:
        raise ValueError("No recordings available for selection.")

    print(f"---\nAvailable DBS recordings: {list(range(n_recordings))}")

    while True:
        try:
            rec_num = input(f"Enter the DBS recording number to read (0-{n_recordings-1}): ").strip()
            
            if not rec_num:
                print("---\nInput cannot be empty. Please enter a valid recording number.")
                continue
            
            rec_num = int(rec_num)
            print(f"---\nReading DBS recording {rec_num}...")

            if 0 <= rec_num < n_recordings:
                return rec_num
            else:
                print("---\nInvalid DBS recording number. Please try again.")
        except ValueError:
            print(f"---\nInvalid input. Please enter a valid DBS recording number (0 - {n_recordings-1}).")


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
    # print length of signal in seconds and samples
    print(f"...length of dbs signal in seconds: {len(df)/fs}")
    print(f"...length of dbs signal in samples: {len(df)}")
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



if __name__ == "__main__":
    file_path = "/Users/lenasalzmann/dev/dbs-eeg-sync/data/eeg_example.set"
    eeg_data = load_eeg_data(file_path)
    print(eeg_data)
