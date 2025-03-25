from data_loader import load_eeg_data, dbs_artifact_settings, open_json_file, select_recording, read_time_domain_data
from source.sync_peaks_finder import find_eeg_peak, find_dbs_peak
from synchronizer import crop_data, synchronize_data, save_synchronized_data

def main():
    """
    Main script to load EEG and DBS data, detect synchronization peaks, align both signals, and save the output.
    """

    # Ask user for file paths

    save_option = input("Testing? (yes/no): ").strip().lower()
    if save_option == "yes":
        eeg_file = "data/eeg_example.set"
        dbs_file = "data/Report_Json_Session_Report_20241025T120701.json"
    else:
        eeg_file = input("Enter EEG file path: ").strip()
        dbs_file = input("Enter DBS file path: ").strip()

    try:
        # Load EEG
        eeg_data = load_eeg_data(eeg_file)
        eeg_fs = eeg_data.info["sfreq"]

        # Find EEG peak
        dbs_freq_min, dbs_freq_max, dbs_duration_sec = dbs_artifact_settings()
        eeg_peak_fs = find_eeg_peak(eeg_data, dbs_freq_min, dbs_freq_max, dbs_duration_sec, save_dir="plots")

        # Load DBS
        json_data = open_json_file(dbs_file)
        block_num = select_recording(json_data)
        dbs_data = read_time_domain_data(json_data, block_num)

        # Find DBS peak
        dbs_peak_fs = find_dbs_peak(dbs_data, save_dir="plots")

        # Synchronize EEG and cropped DBS
        print("Synchronizing EEG and DBS...")
        cropped_eeg, cropped_dbs = crop_data(eeg_data, dbs_data, dbs_peak_fs, eeg_peak_fs)
        print("EEG and DBS cropped at the synchronization peak.")

        # Plot synchronized signals
        synchonized_eeg, synchronized_dbs = synchronize_data(cropped_eeg, cropped_dbs, "plots")

        # Ask user if they want to save the synchronized data
        save_option = input("Save cropped and synchronized EEG & DBS data? (yes/no): ").strip().lower()
        if save_option == "yes":
            save_synchronized_data(synchonized_eeg, synchronized_dbs)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


