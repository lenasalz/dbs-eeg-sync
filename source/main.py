from data_loader import load_eeg_data, open_json_file, select_block, read_time_domain_data
from synchronizer import synchronize_eeg_dbs, find_eeg_peak, find_dbs_peak, plot_synchronized_signals

def main():
    """
    Main script to load EEG and DBS data, detect synchronization peaks, and align both signals.
    """
    eeg_file = input("Enter EEG file path: ").strip()
    dbs_file = input("Enter DBS file path: ").strip()

    try:
        # Load EEG
        eeg_data = load_eeg_data(eeg_file)
        eeg_fs = eeg_data.info["sfreq"]

        # Find EEG peak
        peak_fs, peak_s = find_eeg_peak(eeg_data, 120, 130, 4, duration_sec=120, save_dir="plots")
        print(f"Highest detected EEG peak at {peak_fs} samples ({peak_s:.2f}s)")

        # Load DBS
        json_data = open_json_file(dbs_file)
        block_num = select_block(json_data)
        dbs_data, dbs_fs = read_time_domain_data(json_data, block_num)

        # Find DBS peak
        peak_dbs_idx, peak_dbs_time, cropped_dbs = find_dbs_peak(dbs_data, dbs_fs, save_dir="plots")
        print(f"Highest detected DBS peak at {peak_dbs_idx} samples ({peak_dbs_time:.2f}s)")

        # Synchronize EEG and cropped DBS
        sync_option = input("Synchronize EEG and DBS? (yes/no): ").strip().lower()
        if sync_option == "yes":
            synchronized_eeg, synchronized_dbs = synchronize_eeg_dbs(eeg_data, cropped_dbs, eeg_fs, dbs_fs, peak_fs)
            print("EEG and DBS synchronized and cropped.")

            # Plot synchronized signals
            plot_synchronized_signals(synchronized_eeg, synchronized_dbs, peak_s, save_dir="plots")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()