# sync_peak_finder.py
# Functions for finding synchronization peaks in EEG and DBS data.
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import find_peaks, resample

def find_eeg_peak(raw_eeg, freq_low, freq_high, decim, duration_sec=120, save_dir=None, log_file="sync_log.txt"):
    """
    Finds the highest synchronization peak index in EEG data within a specified frequency band.

    Args:
        raw_eeg (mne.io.Raw): The raw EEG data object to analyze.
        freq_low (float): Lower bound of the frequency range of interest.
        freq_high (float): Upper bound of the frequency range of interest.
        decim (int): Decimation factor for downsampling.
        duration_sec (int, optional): Duration of the cropped EEG window (default: 120 sec).
        save_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
        log_file (str, optional): Path to the log file where peak detection results will be saved.

    Returns:
        int: Peak index in EEG sampling frequency.
        float: Peak time in seconds.
    """

    # Crop and filter EEG
    _raw = raw_eeg.copy()
    raw_cropped = _raw.crop(tmax=duration_sec)
    raw_filtered = raw_cropped.filter(l_freq=freq_low, h_freq=freq_high)

    fs = raw_filtered.info["sfreq"]
    freqs = np.arange(freq_low, freq_high, 1)
    n_cycles = freqs / 1.5  
    time_bandwidth = 4.0  

    power = mne.time_frequency.tfr_multitaper(
        raw_filtered,
        picks="eeg",
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        decim=decim,
        average=True,
        return_itc=False
    )

    power_band_sum = power.data.mean(axis=0).sum(axis=0)  
    time_axis = np.arange(len(power_band_sum)) * decim / fs

    # Detect peaks
    peaks, peak_properties = find_peaks(power_band_sum, height=np.mean(power_band_sum) * 1.5)

    if len(peaks) > 0:
        # Select the peak with the highest power value
        highest_peak_idx = np.argmax(peak_properties["peak_heights"])
        peak_power_idx = peaks[highest_peak_idx]
    else:
        # If no peak is found, use the maximum value in the first 1000 samples as a fallback
        peak_power_idx = np.argmax(power_band_sum[:1000])  

    eeg_peak_index_fs = int(peak_power_idx * decim)
    eeg_peak_index_s = eeg_peak_index_fs / fs

    # Log the detected peak
    with open(log_file, "a") as log:
        log.write(f"Detected highest EEG peak at {eeg_peak_index_fs} samples ({eeg_peak_index_s:.2f} sec)\n")

    print(f"Peak time logged in {log_file}")

    # Plot the detected peak
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, power_band_sum, label="Power in Band (120-130 Hz)")
    plt.axvline(time_axis[peak_power_idx], color='r', linestyle='--', label=f'Highest Peak @ {time_axis[peak_power_idx]:.2f} sec')
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.title('EEG: Power Over Time in 125 Hz Band')
    plt.legend()

    if save_dir:
        plt.savefig(f"{save_dir}/syncPeakEEG.png")
        print(f"Plot saved to {save_dir}/syncPeakEEG.png")

    plt.show()  # Ensures the plot opens

    return eeg_peak_index_fs


def find_dbs_peak(dbs_data, save_dir=None, log_file="sync_log.txt"):
    """
    Finds the highest peak in DBS data in the positive direction only.

    Args:
        dbs_data (pd.DataFrame): The DBS time series data.
        dbs_fs (float): Sampling frequency of the DBS.
        save_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
        log_file (str, optional): Log file path for saving detected peak info.

    Returns:
        int: Peak index in DBS samples.
        float: Peak time in seconds.
        pd.DataFrame: Cropped DBS data from the detected peak onward.
    """

    # Extract DBS signal
    dbs_signal = dbs_data["TimeDomainData"].values
    dbs_fs = dbs_data["SampleRateInHz"][0]

    # Compute time axis
    dbs_time_axis = np.arange(len(dbs_signal)) / dbs_fs

    # Find peaks **only in the positive direction**
    peaks, _ = find_peaks(dbs_signal, height=0)  # Only positive peaks

    if len(peaks) > 0:
        # Select the **highest** positive peak
        dbs_peak_index_fs = peaks[np.argmax(dbs_signal[peaks])]
    else:
        # Fallback: Use max value in the first 1000 samples
        dbs_peak_index_fs = np.argmax(dbs_signal[:1000])

    dbs_peak_index_s = dbs_peak_index_fs / dbs_fs

    # Log detected peak
    with open(log_file, "a") as log:
        log.write(f"Detected DBS peak at {dbs_peak_index_fs} samples ({dbs_peak_index_s:.2f} sec)\n")

    print(f"DBS peak time logged in {log_file}")

    # Plot detected peak
    plt.figure(figsize=(10, 5))
    plt.plot(dbs_time_axis, dbs_signal, label="DBS Signal")
    plt.axvline(dbs_time_axis[dbs_peak_index_fs], color='r', linestyle='--', label=f'Peak @ {dbs_peak_index_s:.2f} sec')
    plt.xlabel('Time (s)')
    plt.ylabel('DBS Amplitude')
    plt.title('DBS Peak Detection')
    plt.legend()

    if save_dir:
        plt.savefig(f"{save_dir}/syncPeakDBS.png")
        print(f"Plot saved to {save_dir}/syncPeakDBS.png")

    plt.show()

    return dbs_peak_index_fs