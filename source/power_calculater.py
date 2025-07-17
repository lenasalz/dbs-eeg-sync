import mne  
import numpy as np
from typing import Tuple, Optional
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import mne


def compute_samplewise_eeg_power(
    eeg_raw: mne.io.Raw,
    freq_low: int,
    freq_high: int,
    channel: str = 'POz',
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
        channel (str): Channel to analyze (default "POz").
        smoothing_sec (float): Smoothing window length in seconds (default 0.5s).
        plot (bool): Whether to plot the power trace.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (power_trace, time_axis)
    """
    
    # Copy, crop, and filter EEG
    _raw = eeg_raw.copy().pick(channel)
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
    power, time = compute_samplewise_eeg_power(eeg_data, 8, 12, channel="T8", plot=True)
    # save power to csv in outputs folder
    np.savetxt("outputs/outputData/eeg_power.csv", power, delimiter=",")
    print(power.shape, time.shape)


    