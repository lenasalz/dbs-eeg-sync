from data_loader import load_eeg_data, dbs_artifact_settings, open_json_file, select_recording, read_time_domain_data
from sync_artefact_finder import detect_dbs_sync_artifact, detect_eeg_sync_artifact, confirm_sync_selection
from synchronizer import cut_data_at_sync, synchronize_data, save_synchronized_data, save_sync_info
from slider import run_manual_sync_slider

import pandas as pd
import sys
import os

print(sys.executable)

def main():
    """
    Main script to load EEG and DBS data, detect synchronization peaks, align both signals, and save the output.
    """ 

    try:
        # Ask user for input and paths
        sub_id = input("Enter subject ID: ").strip()
        block = input("Enter block name: ").strip()
        test_mode = input("---\nTesting? (yes/no): ").strip().lower() == "yes"
        if test_mode:
            eeg_file = "data/eeg_example.set"
            dbs_file = "data/dbs_example.json"
        else:
            eeg_file = input("Enter EEG file path: ").strip()
            dbs_file = input("Enter DBS file path: ").strip()
        
        # Load EEG
        print("---\nLoading EEG data...")
        eeg_data, eeg_fs = load_eeg_data(eeg_file)

        # Load DBS
        print("---\nLoading DBS data...")
        json_data = open_json_file(dbs_file)
        block_num = select_recording(json_data)
        dbs_data = read_time_domain_data(json_data, block_num)

        # Set DBS artifact settings (frequency range)
        dbs_freq_min, dbs_freq_max, _, _ = dbs_artifact_settings()
        print("---\nDetecting EEG sync artifact...")

        # Prompt user for time range of the artifact in the EEG
        max_duration = eeg_data.times[-1]
        print(f"EEG duration: {max_duration:.2f} seconds")

        start_sec = float(input(f"Enter start time (0 to {max_duration:.2f} sec): "))
        end_sec = float(input(f"Enter end time ({start_sec:.2f} to {max_duration:.2f} sec): "))

        if not (0 <= start_sec < end_sec <= max_duration):
            raise ValueError("Invalid time range. Must be within EEG data duration.")
            
        time_range = (start_sec, end_sec)

        channel, eeg_sync_idx, eeg_sync_s, result, smoothed_power = detect_eeg_sync_artifact(eeg_data, freq_low=dbs_freq_min, freq_high=dbs_freq_max, time_range=time_range, plot=True, save_dir='outputs/plots', sub_id=sub_id, block=block)
 
        # Confirm EEG selection
        if channel is not None and eeg_sync_idx is not None:
            confirmed = confirm_sync_selection(channel, eeg_sync_s, result["type"], result["magnitude"])
    
            if confirmed:
                # Proceed with auto-detected sync
                print("✅ Auto sync accepted.")
            else:
                # Trigger manual slider function
                print("❌ Auto sync rejected. Please select the sync artifact manually.")
                smoothed_power_df = pd.DataFrame(smoothed_power)
                selected_eeg_index = run_manual_sync_slider(smoothed_power_df, sub_id, block)
                if selected_eeg_index is not None:
                    eeg_sync_idx = selected_eeg_index
                    eeg_sync_s = selected_eeg_index/eeg_fs
        else:
            raise ValueError("EEG sync detection failed. Cannot proceed.")

        # Find DBS sync artifact
        dbs_signal =  dbs_data["TimeDomainData"].values
        dbs_fs = dbs_data["SampleRateInHz"][0]
        print("---\nPlease close DBS sync plot to continue")
        dbs_sync_idx, dbs_sync_s = detect_dbs_sync_artifact(dbs_signal, dbs_fs, save_dir="outputs/plots", sub_id=sub_id, block=block)

        # Save peak info
        save_sync_info(sub_id, block, eeg_file, dbs_file, eeg_sync_idx, eeg_sync_s, dbs_sync_idx, dbs_sync_s)

        # Synchronize EEG and cropped DBS
        print("---\nSynchronizing EEG and DBS...")
        cropped_eeg, cropped_dbs = cut_data_at_sync(eeg_data, dbs_data, dbs_sync_idx, eeg_sync_idx)
        # Plot synchronized signals
        synchonized_eeg, synchronized_dbs = synchronize_data(cropped_eeg, cropped_dbs, resample_data=None, save_dir="outputs/plots", sub_id=sub_id, block=block)

        # Ask user if they want to save the synchronized datay
        save_option = input("---\nSave cropped and synchronized EEG & DBS data? (yes/no): ").strip().lower()
        if save_option == "yes":
            print("---\nsave_synchronized_data...")
            save_synchronized_data(synchonized_eeg, synchronized_dbs, "outputs/outputData", sub_id, block)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


