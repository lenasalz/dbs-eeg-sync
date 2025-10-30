import mne  
import json
import pandas as pd
import numpy as np
import pyxdf
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


def load_eeg_data(file_path: str | Path):
    """
    Loads EEG data from various file formats using MNE.

    Args:
        file_path (str | Path): Path to the EEG file.

    Returns:
        tuple[mne.io.BaseRaw, float]: The loaded MNE Raw-like object and its sampling frequency.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    # Normalize and coerce to Path
    file_path = Path(str(file_path).strip('\'"')).expanduser().resolve()
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = file_path.suffix.lower()
    
    if ext == ".set":
        raw = mne.io.read_raw_eeglab(str(file_path), preload=True)
    elif ext == ".fif":
        raw = mne.io.read_raw_fif(str(file_path), preload=True)
    elif ext in [".vhdr", ".eeg", ".vmrk"]:
        raw = mne.io.read_raw_brainvision(str(file_path), preload=True)
    elif ext == ".edf":
        raw = mne.io.read_raw_edf(str(file_path), preload=True)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(str(file_path), preload=True)
    elif ext == ".gdf":
        raw = mne.io.read_raw_gdf(str(file_path), preload=True)
    elif ext == ".cnt":
        raw = mne.io.read_raw_cnt(str(file_path), preload=True)
    elif ext == ".mff":
        raw = mne.io.read_raw_egi(str(file_path), preload=True)
    elif ext == ".xdf":
        streams, _ = pyxdf.load_xdf(str(file_path))
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

            # Add annotations for visualization
            durations = [0] * len(marker_labels)
            descriptions = marker_labels.astype(str).tolist()
            annotations = mne.Annotations(onset=marker_sample_indices / sfreq, duration=durations, description=descriptions)
            raw.set_annotations(annotations)
        else:
            logger.warning("No marker stream found in the .xdf file")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    sfreq = raw.info['sfreq']
    logger.info(f"Successfully loaded {file_path}")
    return raw, sfreq


def dbs_artifact_settings(
    freq_min: float | None = None,
    freq_max: float | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    interactive: bool = False
) -> tuple[float, float, float | None, float | None]:
    """
    Frequency range and time range settings for DBS artifact detection.
    
    Args:
        freq_min: Minimum frequency (Hz). Defaults to 120.0.
        freq_max: Maximum frequency (Hz). Defaults to 130.0.
        tmin: Start time (seconds). Defaults to None (full range).
        tmax: End time (seconds). Defaults to None (full range).
        interactive: Deprecated parameter (ignored). Always uses non-interactive mode.
    
    Returns:
        tuple[float, float, float | None, float | None]: (freq_min, freq_max, tmin, tmax)
    
    Examples:
        >>> fmin, fmax, tmin, tmax = dbs_artifact_settings(freq_min=115.0, freq_max=135.0)
        >>> print(fmin, fmax)
        115.0 135.0
    """
    fmin = 120.0 if freq_min is None else float(freq_min)
    fmax = 130.0 if freq_max is None else float(freq_max)
    return fmin, fmax, tmin, tmax


def open_json_file(filepath: str | Path) -> dict:
    """
    Opens a JSON file and returns its contents as a dictionary.

    Args:
        filepath (str | Path): Path to the JSON file.

    Returns:
        dict: Dictionary containing the JSON data.
    """
    p = Path(str(filepath).strip('\'"')).expanduser().resolve()
    with p.open() as jsonfile:
        data = json.load(jsonfile)
    return data


def select_recording(json_data: dict, index: int | None = None) -> int:
    """
    Select a DBS recording from the JSON data.
    
    Args:
        json_data: Parsed DBS JSON data containing "BrainSenseTimeDomain" key.
        index: Recording index to select (0-based). If None, defaults to 0.
    
    Returns:
        int: Selected recording index.
        
    Raises:
        ValueError: If no recordings are available or index is out of range.
    
    Examples:
        >>> json_data = {"BrainSenseTimeDomain": [{...}, {...}]}
        >>> rec_num = select_recording(json_data, index=0)
        >>> print(rec_num)
        0
    """
    recordings = json_data.get("BrainSenseTimeDomain", [])
    n_recordings = len(recordings)

    if n_recordings == 0:
        raise ValueError("No recordings available in BrainSenseTimeDomain.")
    
    # Default to first recording if index not provided
    if index is None:
        logger.info(f"No recording index specified, using first recording (0 of {n_recordings} available)")
        return 0
    
    # Validate index
    idx = int(index)
    if not (0 <= idx < n_recordings):
        raise ValueError(
            f"Recording index {idx} out of range. "
            f"Available recordings: 0-{n_recordings-1}"
        )
    
    return idx


def read_time_domain_data(json_data: dict, rec_num: int) -> pd.DataFrame:
    """
    Extracts and returns TimeDomainData from the selected recording.

    Args:
        json_data: Parsed JSON data containing "BrainSenseTimeDomain" key.
        rec_num: Selected recording number (0-based index).

    Returns:
        pd.DataFrame: DataFrame with columns ["TimeDomainData", "recording", "SampleRateInHz"].
    """
    fs = json_data["BrainSenseTimeDomain"][rec_num].get("SampleRateInHz")
    
    df = pd.DataFrame(json_data["BrainSenseTimeDomain"][rec_num]["TimeDomainData"], columns=["TimeDomainData"])
    df["recording"] = rec_num
    df["SampleRateInHz"] = fs
    logger.info(f"Successfully read DBS recording {rec_num} with sampling frequency {fs:.2f} Hz")
    logger.debug(f"Length of DBS signal: {len(df)/fs:.2f}s ({len(df)} samples)")
    return df


def read_lfp_data(json_data: dict, rec_num: int) -> tuple[pd.DataFrame, float, str]:
    """
    Extracts and returns LfpData from the selected block.

    Args:
        json_data: Parsed JSON data containing "BrainSenseLfp" key.
        rec_num: Selected recording number (0-based index).

    Returns:
        tuple[pd.DataFrame, float, str]: (DataFrame with LFP data, sampling frequency in Hz, lead side).
    """
    fs = json_data["BrainSenseLfp"][rec_num].get("SampleRateInHz")
    channel = json_data["BrainSenseLfp"][rec_num]["Channel"]
    lead = "Left" if "LEFT" in channel else "Right" if "RIGHT" in channel else "Unknown"
    
    lfp_data = [sample[lead].get("LFP") for sample in json_data["BrainSenseLfp"][rec_num]["LfpData"]]
    df = pd.DataFrame(lfp_data, columns=["LfpData"])
    df["recording"] = rec_num
    logger.info(f"Successfully read LFP recording {rec_num} with sampling frequency {fs:.2f} Hz")
    return df, fs, lead



if __name__ == "__main__":
    # Use relative path from repository root
    test_path = Path(__file__).parent.parent / "data" / "eeg_example.set"
    raw, fs = load_eeg_data(test_path)
    print(f"EEG loaded in __main__: fs={fs:.3f} Hz, n_channels={len(raw.ch_names)}, n_times={raw.n_times}")
