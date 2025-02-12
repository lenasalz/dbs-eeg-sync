	# •	Inside pipeline.py (or main.py), you should:
	# •	Import functions from data_loader.py, synchronizer.py, etc.
	# •	Process input (e.g., file paths or configurations).
	# •	Run the entire workflow.


# main.py

import sys
from data_loader import load_eeg_data, open_json_file, select_block, read_time_domain_data

def main():
    """
    Main script to load EEG or DBS data.
    """
    if len(sys.argv) < 3:
        file_type = input("Enter file type (EEG or DBS): ").strip().lower()
        file_path = input("Enter the file path: ").strip()
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
            df, fs = read_time_domain_data(json_data, block_num)
            print(f"Loaded TimeDomainData from block {block_num} with sampling frequency {fs} Hz")
            print(df.head())

        else:
            print("Invalid file type. Use 'EEG' or 'DBS'.")
    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    main()