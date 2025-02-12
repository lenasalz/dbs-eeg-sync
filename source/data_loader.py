import os
import numpy as np
import mne  # Assuming EEG data uses MNE library
import json
import pandas as pd
import sys

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
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    print(f"Successfully loaded {file_path}")
    return raw

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

def select_block(json_data: dict) -> int:
    """
    Prompts the user to select a block from the JSON data.

    Args:
        json_data (dict): Parsed JSON data.

    Returns:
        int: Selected block number.
    
    Raises:
        ValueError: If no blocks are available for selection.
    """
    blocks = json_data.get("BrainSenseTimeDomain", [])
    n_blocks = len(blocks)

    if n_blocks == 0:
        raise ValueError("No blocks available for selection.")

    print(f"Available blocks: {list(range(n_blocks))}")

    while True:
        try:
            block_num = input(f"Enter the block number to read (0-{n_blocks-1}): ").strip()
            
            if not block_num:
                print("Input cannot be empty. Please enter a valid block number.")
                continue
            
            block_num = int(block_num)
            if 0 <= block_num < n_blocks:
                return block_num
            else:
                print("Invalid block number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid block number.")

def read_time_domain_data(json_data: dict, block_num: int) -> tuple:
    """
    Extracts and returns TimeDomainData from the selected block.

    Args:
        json_data (dict): Parsed JSON data.
        block_num (int): Selected block number.

    Returns:
        tuple: (DataFrame containing the time domain data, sampling frequency in Hz).
    """
    fs = json_data["BrainSenseTimeDomain"][block_num].get("SampleRateInHz")
    
    df = pd.DataFrame(json_data["BrainSenseTimeDomain"][block_num]["TimeDomainData"], columns=["TimeDomainData"])
    df["block"] = block_num
    return df, fs

def read_lfp_data(json_data: dict, block_num: int) -> tuple:
    """
    Extracts and returns LfpData from the selected block.

    Args:
        json_data (dict): Parsed JSON data.
        block_num (int): Selected block number.

    Returns:
        tuple: (DataFrame containing the LFP data, sampling frequency in Hz, lead side).
    """
    fs = json_data["BrainSenseLfp"][block_num].get("SampleRateInHz")
    channel = json_data["BrainSenseLfp"][block_num]["Channel"]
    lead = "Left" if "LEFT" in channel else "Right" if "RIGHT" in channel else "Unknown"
    
    lfp_data = [sample[lead].get("LFP") for sample in json_data["BrainSenseLfp"][block_num]["LfpData"]]
    df = pd.DataFrame(lfp_data, columns=["LfpData"])
    df["block"] = block_num
    return df, fs, lead

def main():
    """
    Main function to load either EEG or DBS data based on user input or command-line arguments.
    """
    if len(sys.argv) < 3:
        file_type = input("Enter the file type to load (EEG or DBS): ").strip().lower()
        file_path = input("Enter the path to the file: ").strip()
    else:
        file_type = sys.argv[1].lower()
        file_path = sys.argv[2]

    try:
        if file_type == "eeg":
            eeg_data = load_eeg_data(file_path)
            print(f"EEG Data Loaded: {eeg_data}")
            
        elif file_type == "dbs":
            json_data = open_json_file(file_path)
            block_num = select_block(json_data)
            
            data_type = input("Enter data type to read (TimeDomainData or LfpData): ").strip()
            if data_type.lower() == "timedomaindata":
                df, fs = read_time_domain_data(json_data, block_num)
                print(f"Loaded TimeDomainData from block {block_num} with sampling frequency {fs} Hz")
            elif data_type.lower() == "lfpdata":
                df, fs, lead = read_lfp_data(json_data, block_num)
                print(f"Loaded LfpData from block {block_num} on {lead} lead with sampling frequency {fs} Hz")
            else:
                print("Invalid data type selected. Exiting.")
                return

            print(df.head())
        else:
            print("Invalid file type. Please enter 'EEG' or 'DBS'.")
    except Exception as e:
        print(f"Error loading {file_type.upper()} data: {e}")

if __name__ == "__main__":
    main()
