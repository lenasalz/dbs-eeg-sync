import mne  
import numpy as np
from typing import Tuple, Optional
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


def compute_eeg_power(
    raw_eeg: mne.io.Raw, 
    freq_low: int, 
    freq_high: int, 
    duration_sec: int=120, 
    channel: str="POz",
    plot=False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Computes the power of EEG data in a specified frequency range and duration.

    Args:
        raw_eeg (mne.io.Raw): The raw EEG data.
        freq_low (int): Lower frequency bound for power computation.
        freq_high (int): Upper frequency bound for power computation.
        duration_sec (int, optional): Duration in seconds for which to compute the power. Defauls to 120 seconds.
        channel (str, optional): The EEG channel to analyze. Defaults to "Cz".

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing:
            - eeg_power_band_sum: The computed power in the specified frequency range. Caveat: The power is averaged across all channels if channel is None. 
                Also, the resulting power is lower-resolution power trace, not sample-wise EEG power.
            - power_time_axis: Time axis corresponding to the computed power.
    """

    # Crop and filter EEG
    _raw = raw_eeg.copy()
    raw_cropped = _raw.crop(tmax=duration_sec)
    raw_filtered = raw_cropped.filter(l_freq=freq_low, h_freq=freq_high, verbose='ERROR')

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
    # if channel is given, use it, otherwise use all channels
    if channel is None:
        picks = "eeg"
    else:
        picks = channel    # only one channel is used for the analysis

    power = raw_filtered.compute_tfr(
        method='multitaper',
        picks=picks,
        freqs=freqs,
        n_cycles=n_cycles,
        time_bandwidth=time_bandwidth,
        decim=1,
    )
    eeg_power_band_sum = power.data.mean(axis=0).sum(axis=0)  

    power_time_axis = power.times
    if plot:
        # plot the power
        plt.figure()
        plt.plot(power.times, eeg_power_band_sum)
        plt.title(f"Power of {picks} in {freq_low}-{freq_high} Hz")
        plt.xlabel("Time (s)")
        plt.ylabel("Power")
        plt.show()

    return eeg_power_band_sum, power_time_axis


def compute_samplewise_eeg_power(
    raw_eeg: mne.io.Raw,
    freq_low: int,
    freq_high: int,
    duration_sec: int = 120,
    channel: str = "POz",
    smoothing_sec: Optional[float] = 0.5,
    plot: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes sample-wise band power of EEG data in a specified frequency range, with optional smoothing. 
    The sample-wise power is computed by averaging the power of the EEG data in the specified frequency range.
    The resulting power trace is of the same length as the EEG data.

    Args:
        raw_eeg (mne.io.Raw): The raw EEG data.
        freq_low (int): Lower frequency bound.
        freq_high (int): Upper frequency bound.
        duration_sec (int): Duration in seconds (default 120).
        channel (str): Channel to analyze (default "POz").
        smoothing_sec (float): Smoothing window length in seconds (default 0.5s).
        plot (bool): Whether to plot the power trace.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (power_trace, time_axis)
    """
    # Copy, crop, and filter EEG
    _raw = raw_eeg.copy().pick(channel).crop(tmax=duration_sec)
    fs = _raw.info["sfreq"]
    _raw.filter(l_freq=freq_low, h_freq=freq_high, verbose='ERROR')

    # Get data
    data = _raw.get_data().squeeze()  # shape (n_samples,)

    # Hilbert transform to get analytic signal
    analytic_signal = hilbert(data)
    power_trace = np.abs(analytic_signal) ** 2

    # Optional smoothing
    if smoothing_sec and smoothing_sec > 0:
        smoothing_samples = int(smoothing_sec * fs)
        power_trace = uniform_filter1d(power_trace, size=smoothing_samples)

    # Time axis
    time_axis = np.arange(len(power_trace)) / fs

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, power_trace, label=f'{freq_low}-{freq_high} Hz Power')
        plt.xlabel("Time (s)")
        plt.ylabel("Power")
        plt.title(f"Sample-wise Band Power ({channel}) - Smoothed ({smoothing_sec}s)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return power_trace, time_axis

from source.data_loader import load_eeg_data

if __name__ == '__main__':
    file_path = "/Users/lenasalzmann/dev/dbs-eeg-sync/data/eeg_example.set"
    eeg_data = load_eeg_data(file_path)
    power, time = compute_samplewise_eeg_power(eeg_data, 8, 12, channel="POz", plot=True)
    print(power.shape, time.shape)


    