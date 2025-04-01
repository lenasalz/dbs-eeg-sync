# sync_peak_finder.py
# Functions for finding synchronization peaks in EEG and DBS data.
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import csv

def find_eeg_peak(raw_eeg, freq_low, freq_high, decim=1, duration_sec=120, save_dir=None, log_file="sync_log.txt"):

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
    picks = "C3"    # only one channel is used for the analysis

    power = mne.time_frequency.tfr_multitaper(
        raw_filtered,
        # picks="eeg",
        picks=picks,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        decim=decim,
        average=True,
        return_itc=False
    )

    eeg_power_band_sum = power.data.mean(axis=0).sum(axis=0)  

    # for the peak detection, ignore the first and last second, as there are artefacts from power computation
    cut_samples = int((fs / decim) * 1)
    eeg_power_band_sum = eeg_power_band_sum[cut_samples:-cut_samples]
    
    time_axis = np.arange(len(eeg_power_band_sum)) * decim / fs
   
    # Detect peaks
    peaks, peak_properties = find_peaks(eeg_power_band_sum, height=np.mean(eeg_power_band_sum) * 1.5)

    if len(peaks) > 0:
        # Select the peak with the highest power value
        highest_peak_idx = np.argmax(peak_properties["peak_heights"])
        peak_power_idx = peaks[highest_peak_idx]
    else:
        # If no peak is found, use the maximum value in the first 1000 samples as a fallback
        peak_power_idx = np.argmax(eeg_power_band_sum[:1000])  

    eeg_peak_index_fs = int(peak_power_idx * decim)
    eeg_peak_index_s = eeg_peak_index_fs / fs

    # Log the detected peak
    with open(log_file, "a") as log:
        log.write(f"Detected highest EEG peak at {eeg_peak_index_fs} samples ({eeg_peak_index_s:.2f} sec)\n")

    print(f"---\nPeak time logged in {log_file}")

    # Plot the detected peak
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, eeg_power_band_sum, label="Power in Band (120-130 Hz)")
    plt.axvline(time_axis[peak_power_idx], color='r', linestyle='--', label=f'Highest Peak @ {time_axis[peak_power_idx]:.2f} sec')
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.title('EEG: Power Over Time in 125 Hz Band')
    plt.legend()
    
    if save_dir:
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/syncPeakEEG_{dat}.png")
        print(f"---\nPlot saved to {save_dir}/syncPeakEEG_{dat}.png")
    
    print("---\nPlease close the plot to continue.\n---")

    plt.show()  # Ensures the plot opens

    return eeg_peak_index_fs, eeg_peak_index_s, eeg_power_band_sum


def find_dbs_peak(dbs_signal, dbs_fs, save_dir=None, log_file="sync_log.txt"):
    """
    Finds the highest peak in DBS data in the positive direction only.

    Args:
        dbs_signal (): The DBS time series data.
        dbs_fs (float): Sampling frequency of the DBS.
        save_dir (str, optional): Directory to save the plot. If None, the plot is not saved.
        log_file (str, optional): Log file path for saving detected peak info.

    Returns:
        int: Peak index in DBS samples.
        float: Peak time in seconds.
        pd.DataFrame: Cropped DBS data from the detected peak onward.
    """

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

    print(f"---\nDBS peak time logged in {log_file}")

    # Plot detected peak
    plt.figure(figsize=(10, 5))
    plt.plot(dbs_time_axis, dbs_signal, label="DBS Signal")
    plt.axvline(dbs_time_axis[dbs_peak_index_fs], color='r', linestyle='--', label=f'Peak @ {dbs_peak_index_s:.2f} sec')
    plt.xlabel('Time (s)')
    plt.ylabel('DBS Amplitude')
    plt.title('DBS Peak Detection')
    plt.legend()

    if save_dir:
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_dir}/syncPeakDBS_{dat}.png")
        print(f"---\nPlot saved to {save_dir}/syncPeakDBS_{dat}.png")
    
    plt.show()

    return dbs_peak_index_fs, dbs_peak_index_s


def save_sync_peak_info(eeg_file, dbs_file, eeg_peak_idx, eeg_peak_time, dbs_peak_idx, dbs_peak_time, output_file="sync_info.csv"):
    """
    Save EEG & DBS peak indices and times with file names to a CSV table.
    """
    header = ["EEG File", "DBS File", "EEG Peak Index", "EEG Peak Time (s)", "DBS Peak Index", "DBS Peak Time (s)"]
    row = [eeg_file, dbs_file, eeg_peak_idx, eeg_peak_time, dbs_peak_idx, dbs_peak_time]

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{output_file}"
    
    try:
        write_header = not os.path.isfile(output_path)

        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
        print(f"---\nPeak info saved to {output_path}")

    except Exception as e:
        print(f"Error saving peak info: {e}")


def detect_eeg_drop_onset(raw_eeg, freq_low, freq_high, decim=1, cut_sec=1, smooth_window=301, threshold_factor=1, plot=False, save_dir=None):
    """
    Detect the onset of a drop in power_band_sum based on gradient.

    Parameters:
    - raw_eeg: mne.io.Raw
        The raw EEG data object to analyze.
    - freq_low: float
        Lower bound of the frequency range of interest.
    - freq_high: float
        Upper bound of the frequency range of interest.
    - decim: int
        Decimation factor used in power computation.
    - cut_sec: float
        Seconds to exclude at start and end.
    - smooth_window: int
        Window length for Savitzky-Golay smoothing.
    - threshold_factor: float
        Multiplier for gradient threshold.
    - plot: bool
        If True, plot the detection.
    - save_dir: str or None
        Directory to save plot if desired.

    Returns:
    - drop_onset_idx: int
        Index of drop onset in power_band_sum.
    """

    raw_cropped = raw_eeg.copy().crop(tmax=120)  # crop to first 2 min, where the artifact should be present
    # Filter the raw data to remove low-frequency drifts and high-frequency noise
    raw_filtered = raw_cropped.copy().filter(l_freq=freq_low, h_freq=freq_high)  # Bandpass filter to the 120-130 Hz band
    # Get the sampling frequency
    fs = raw_filtered.info["sfreq"]
    # Define frequencies and wavelet parameters
    freqs = np.arange(freq_low, freq_high, 1)
    n_cycles = freqs / 1.5  # Use a balance between time and frequency resolution
    time_bandwidth = 4.0  # Adjust for a balance between time and frequency resolution
    picks = 'C3'
    # Compute TFR using multitaper
    power = mne.time_frequency.tfr_multitaper(
        raw_filtered,
        # picks="eeg",
        picks=picks,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        n_jobs=1,  # Use parallel processing if supported
        decim=decim,  # Downsampling factor
        average=True,  # Average over epochs
        return_itc=False  # Do not compute inter-trial coherence
    )
    frequencies = power.freqs  # number of frequency rows
    # Extract the power values for the 125 Hz band (approximately 120-130 Hz)
    freq_band_mask = (frequencies >= 120) & (frequencies <= 130)
    # Extract the power values for the 125 Hz band (approximately 120-130 Hz)
    power_band = power.data[:, freq_band_mask, :]

    # Sum power across the selected band to create a time series
    power_band_sum = np.mean(power_band.sum(axis=0), axis=0)

    cut_samples = int((fs / decim) * cut_sec)

    # Smoothing
    smoothed = savgol_filter(power_band_sum, window_length=smooth_window, polyorder=3)

    # Gradient
    gradient = np.gradient(smoothed)

    # Threshold
    search_gradient = gradient[cut_samples:-cut_samples]
    threshold = np.mean(gradient) - threshold_factor * np.std(gradient)
    drop_onset_relative = np.where(search_gradient < threshold)[0][0]
    drop_onset_idx = drop_onset_relative + cut_samples

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(smoothed, label="Smoothed Power Band Sum")
        plt.axvline(drop_onset_idx, color="r", linestyle="--", label="Drop Onset")
        plt.xlabel("Samples")
        plt.ylabel("Power")
        plt.title("Drop Onset Detection via Gradient")
        plt.legend()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "drop_onset_detection.png"))
        plt.show()

    return drop_onset_idx