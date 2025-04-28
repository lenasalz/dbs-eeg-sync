from data_loader import load_eeg_data, dbs_artifact_settings, open_json_file, select_recording, read_time_domain_data
from sync_artefact_finder import find_dbs_peak, save_sync_peak_info, detect_sync_from_eeg, confirm_sync_selection
from synchronizer import crop_data, synchronize_data, save_synchronized_data
from slider import run_manual_sync_slider

import pandas as pd
import sys

print(sys.executable)

def main():
    """
    Main script to load EEG and DBS data, detect synchronization peaks, align both signals, and save the output.
    """ 

    try:
        # Ask user for file paths
        test_mode = input("---\nTesting? (yes/no): ").strip().lower() == "yes"
        if test_mode:
            eeg_file = "data/eeg_example.set"
            dbs_file = "data/dbs_example.json"
        else:
            eeg_file = input("Enter EEG file path: ").strip()
            dbs_file = input("Enter DBS file path: ").strip()
        
        # Load EEG
        eeg_data = load_eeg_data(eeg_file)

        # Load DBS
        json_data = open_json_file(dbs_file)
        block_num = select_recording(json_data)
        dbs_data = read_time_domain_data(json_data, block_num)
        # dbs_data = read_lfp_data(dbs_data, block_num)

        # Find EEG peak
        dbs_freq_min, dbs_freq_max, dbs_duration_sec = dbs_artifact_settings()
        print("---\nDetecting EEG change window...")
        channel, eeg_sync_idx, eeg_sync_s, result, smoothed_power = detect_sync_from_eeg(eeg_data, dbs_freq_min, dbs_freq_max, dbs_duration_sec, plot=True)
 
        # Confirm EEG selection
        if channel is not None and eeg_sync_idx is not None:
            confirmed = confirm_sync_selection(channel, eeg_sync_s, result["type"], result["magnitude"])
            
            if confirmed:
                # Proceed with auto-detected sync
                print("✅ Auto sync accepted.")
            else:
                # Trigger manual slider function
                # enter sub_id and block
                sub_id = input("Enter subject ID: ").strip()
                block = input("Enter block name: ").strip()
                print("❌ Auto sync rejected. Please select the sync artifact manually.")
                smoothed_power_df = pd.DataFrame(smoothed_power)
                run_manual_sync_slider(smoothed_power_df, sub_id, block)

        # Find DBS peak
        dbs_signal =  dbs_data["TimeDomainData"].values
        dbs_fs = dbs_data["SampleRateInHz"][0]
        dbs_peak_idx, dbs_peak_s = find_dbs_peak(dbs_signal, dbs_fs, save_dir="outputs/plots")

        # Save peak info
        save_sync_peak_info(eeg_file, dbs_file, eeg_sync_idx, eeg_sync_s, dbs_peak_idx, dbs_peak_s)

        # Synchronize EEG and cropped DBS
        print("---\nSynchronizing EEG and DBS...")
        cropped_eeg, cropped_dbs = crop_data(eeg_data, dbs_data, dbs_peak_idx, eeg_sync_idx)
        # Plot synchronized signals
        synchonized_eeg, synchronized_dbs = synchronize_data(cropped_eeg, cropped_dbs, "outputs/plots")

        # Ask user if they want to save the synchronized datay
        save_option = input("---\nSave cropped and synchronized EEG & DBS data? (yes/no): ").strip().lower()
        if save_option == "yes":
            save_synchronized_data(synchonized_eeg, synchronized_dbs)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


