# sync_artifact_finder.py
# Functions for finding synchronization peaks in EEG and DBS data.
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d
from datetime import datetime
import os
import logging

from dbs_eeg_sync.power_calculator import compute_samplewise_eeg_power

logger = logging.getLogger(__name__)


def detect_dbs_sync_artifact(
    dbs_signal,
    dbs_fs,
    save_dir: str | None = None,
    sub_id: str | None = None,
    block: str | None = None,
    plot: bool = False,
    **kwargs,
) -> tuple[int, float]:
    """
    Finds the highest positive peak in DBS data.

    Args:
        dbs_signal (np.ndarray): The DBS time series data.
        dbs_fs (int | float): Sampling frequency of the DBS.
        save_dir (Optional[str]): If provided, save a PNG figure here; if None, do not save.
        sub_id (Optional[str]): The subject ID (used for filename/title only).
        block (Optional[str]): The block name (used for filename/title only).
        plot (bool): If True, call plt.show() (interactive). If False, do not show (headless-friendly).

    Returns:
        tuple[int, float]: (dbs_peak_index_fs, dbs_peak_index_s)
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
    # Plot detected peak
    plt.figure(figsize=(10, 5))
    plt.plot(dbs_time_axis, dbs_signal, label="STN-LFP Signal")
    plt.axvline(dbs_time_axis[dbs_peak_index_fs], color='r', linestyle='--', label=f'Artifact @ {dbs_peak_index_s:.2f} sec | {dbs_peak_index_fs} samples')
    plt.xlabel('Time (s)')
    plt.ylabel('STN-LFP Amplitude')
    plt.title(f'DBS Artifact Detection - {sub_id} | {block}')
    plt.legend()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(save_dir, f"{dat}_syncDBS_{sub_id}_{block}.png")
        plt.savefig(out_path)
        logger.info(f"Plot saved to {out_path}")
    
    if plot:
        plt.show()
    else:
        plt.close()

    return dbs_peak_index_fs, dbs_peak_index_s


def detect_eeg_sync_artifact(
    eeg_raw: mne.io.Raw,
    freq_low: float,
    freq_high: float,
    time_range: tuple[float, float] | None = None,
    eeg_fs: int | None = None,
    channel_list: list[str] | None = None,
    smooth_window: int = 301,
    window_size_sec: int = 2,
    plot: bool = False,
    save_dir: str | None = "outputs/plots",
    sub_id: str | None = None,
    block: str | None = None
) -> tuple[
    str | None,
    int | None,
    float | None,
    dict | None,
    np.ndarray | None,
]:
    """
    Detects the sync artifact from the EEG data.

    Args:
        eeg_raw (mne.io.Raw): The EEG data.
        freq_low (float): The lower frequency of the power band.
        freq_high (float): The higher frequency of the power band.
        time_range (Tuple[float, float], optional): (start, stop) time range in seconds to crop for faster processing.
        eeg_fs (Optional[int]): Sampling frequency of the EEG data. If None, uses `eeg_raw.info['sfreq']`.
        channel_list (Optional[List[str]]): Channels to search. If None, uses a default list.
        smooth_window (int): Window size for Savitzky–Golay smoothing of the band-power (samples).
        window_size_sec (int): Window (seconds) for the pre/post mean comparison.
        plot (bool): If True, build a Plotly figure and (optionally) save it.
        save_dir (Optional[str]): Directory to save the plot (PNG) and CSV. If None, do not save.
        sub_id (Optional[str]): Subject ID (for plot/filenames only).
        block (Optional[str]): Block name (for plot/filenames only).

    Returns:
        Tuple[
            Optional[str],           # best_channel
            Optional[int],           # onset_index_global (samples, relative to FULL recording)
            Optional[float],         # onset_time_global (seconds, relative to FULL recording)
            Optional[dict],          # best_result (metadata; indices/times are GLOBAL)
            Optional[np.ndarray],    # best_power (per-sample band power for the best channel, cropped to time_range)
        ]

    Notes:
        * All indices and times in the return values AND in `best_result` are **GLOBAL** (relative to the full recording),
          even if a `time_range` crop was used.
        * For large signals, the returned Plotly figure decimates the trace to ~100k points for responsiveness.
        * If plot=True: may call plt.show(); otherwise save or skip. 
        * Always plt.close(fig) before returning.
    """
    
    if eeg_fs is None:
        eeg_fs = int(eeg_raw.info['sfreq'])

    # --- Validate/clamp frequency band to be below Nyquist ---
    nyq = 0.5 * eeg_fs
    # basic sanitization
    if freq_low < 0:
        freq_low = 0.0
    if freq_high <= 0:
        freq_high = nyq - 1.0
    # clamp high just below Nyquist to avoid MNE filter errors
    if freq_high >= nyq:
        logger.warning(f"freq_high ({freq_high:.3f}) ≥ Nyquist ({nyq:.3f}); clamping to {nyq - 1.0:.3f} Hz")
        freq_high = nyq - 1.0
    # ensure ordering
    if freq_low >= freq_high:
        # widen minimally to keep a valid passband
        freq_low = max(0.0, freq_high - 1.0)
        logger.warning(f"Adjusted freq_low to {freq_low:.3f} Hz to keep freq_low < freq_high ({freq_high:.3f} Hz)")

    # Determine crop range and global sample offset (robust to None/partial ranges)
    last_time = float(eeg_raw.times[-1])
    if time_range is None:
        start_time, stop_time = 0.0, last_time
    else:
        start_time, stop_time = time_range
        # Interpret Nones as full-range endpoints
        if start_time is None:
            start_time = 0.0
        if stop_time is None:
            stop_time = last_time
    # Clamp to valid bounds and ensure order
    start_time = max(0.0, float(start_time))
    stop_time = min(last_time, float(stop_time))
    if stop_time <= start_time:
        # Fallback to full range if invalid
        logger.warning(f"Invalid time_range {time_range}; using full range 0–{last_time:.3f}s")
        start_time, stop_time = 0.0, last_time

    crop_start_sample = int(round(start_time * eeg_fs))

    # Work on a cropped copy for speed, but keep the global offset
    eeg_raw = eeg_raw.copy().crop(tmin=start_time, tmax=stop_time)
    logger.info(f"EEG data cropped: {start_time:.3f}s → {stop_time:.3f}s (Δ={stop_time - start_time:.1f}s)")

    if channel_list is None:
        channel_list = ['Pz', 'Oz', 'Fz', 'Cz', 'T7', 'T8', 'O1', 'O2']

    # Limit channels to those available
    available_channels = eeg_raw.ch_names
    channel_list = [ch for ch in channel_list if ch in available_channels]
    if len(channel_list) == 0:
        logger.warning("No valid channels found in the EEG data.")
        logger.info(f"Available channels: {available_channels}")
        return None, None, None, None, None

    best_channel = None
    best_result = None
    best_power = None
    _channel_scores = []  # (abs_magnitude, channel, result_dict)

    for ch in channel_list:
        try:
            eeg_power, power_time = compute_samplewise_eeg_power(
                eeg_raw,
                freq_low=freq_low,
                freq_high=freq_high,
                channel=ch
            )
        except Exception as e:
            logger.warning(f"Skipping invalid channel {ch}: {e}")
            continue

        # Sanity check
        if len(power_time) != len(eeg_power):
            logger.warning(f"Power time and EEG power length mismatch for channel {ch} (time={len(power_time)}, power={len(eeg_power)})")
            continue

        # --- Robust smoothing: pick a valid Savitzky–Golay window, fallback if needed ---
        polyorder = 3
        n = len(eeg_power)
        # smallest odd window strictly greater than polyorder
        min_wl = polyorder + 2
        if min_wl % 2 == 0:
            min_wl += 1
        # largest odd window not exceeding n
        max_wl = n if (n % 2 == 1) else (n - 1)

        if max_wl >= min_wl:
            wl = min(smooth_window, max_wl)
            if wl % 2 == 0:
                wl -= 1
            wl = max(wl, min_wl)
            smoothed = savgol_filter(eeg_power, window_length=wl, polyorder=polyorder)
        else:
            # too short for SG; use light uniform smoothing or none
            if n >= 5:
                fallback = min(5, n)
                smoothed = uniform_filter1d(eeg_power, size=fallback)
            else:
                smoothed = eeg_power.astype(float)
            logger.warning(f"Savitzky–Golay smoothing skipped on channel {ch}: signal too short (n={n})")
        window_size = int(window_size_sec * eeg_fs)
        if 2 * window_size >= len(smoothed):
            logger.warning(f"Signal too short for windowed diff on channel {ch}")
            continue

        # Windowed mean diff across the smoothed signal
        diffs = []
        for i in range(0, len(smoothed) - 2 * window_size):
            pre = smoothed[i : i + window_size]
            post = smoothed[i + window_size : i + 2 * window_size]
            diff = float(np.mean(post) - np.mean(pre))
            diffs.append((i + window_size, diff))
        if not diffs:
            continue

        idx_local, val = max(diffs, key=lambda x: abs(x[1]))  # index within CROPPED segment

        # Onset refinement via gradient on the smoothed power
        grad = np.gradient(smoothed)
        search_window = int(0.5 * eeg_fs)  # look back 0.5 s
        start_idx = max(idx_local - search_window, 0)
        end_idx = idx_local
        local_max_grad = np.max(np.abs(grad[start_idx:end_idx])) if end_idx > start_idx else 0.0
        slope_threshold = 0.3 * local_max_grad if local_max_grad > 0 else 0.0
        onset_candidates = np.where(np.abs(grad[start_idx:end_idx]) > slope_threshold)[0]
        if len(onset_candidates) > 0:
            onset_local = start_idx + int(onset_candidates[0])
        else:
            onset_local = idx_local

        # Convert to GLOBAL indices/times (relative to the full recording)
        idx_global = crop_start_sample + idx_local
        time_global = start_time + idx_local / eeg_fs
        onset_index_global = crop_start_sample + onset_local
        onset_time_global = start_time + onset_local / eeg_fs

        result = {
            "sub_id": sub_id,
            "block": block,
            "type": "drop" if val < 0 else "spike",
            # Local (cropped) values retained for debugging
            "index_local": idx_local,
            "onset_index_local": onset_local,
            # Global (FULL recording) values used downstream
            "index": idx_global,
            "time": time_global,
            "onset_index": onset_index_global,
            "onset_time": onset_time_global,
            "magnitude": float(val),
            "channel": ch,
            "channel_idx": eeg_raw.ch_names.index(ch),
            "power_time": power_time,  # still cropped time axis
        }
        _channel_scores.append((abs(result["magnitude"]), ch, result))

        if best_result is None or abs(result["magnitude"]) > abs(best_result["magnitude"]):
            best_channel = ch
            best_result = result
            best_power = eeg_power

    if _channel_scores:
        _channel_scores.sort(key=lambda x: x[0], reverse=True)
        top_preview = ", ".join([f"{c}:{m:.3g}" for m, c, _ in _channel_scores[:5]])
        # Only print the heavy hint in interactive mode
        if plot:
            logger.debug(f"Top channels by |Δpower|: {top_preview}")
            logger.info("Close EEG sync plot to continue")

    # Plot once for the best channel only
    if plot and best_result is not None and best_power is not None:
        # Ensure x-axis spans the exact crop window
        time_vector_global = np.linspace(start_time, stop_time, len(best_power), endpoint=False)

        plt.figure(figsize=(10, 4))
        plt.plot(time_vector_global, best_power, label=f"Power {best_channel}")

        event_time = best_result["time"]
        plt.axvline(
            event_time,
            color="red" if best_result["type"] == "drop" else "green",
            linestyle="--",
            label=f"{best_result['type']} at {event_time:.2f}s"
        )

        plt.title(
            f"EEG Sync Detection - {best_channel} | "
            f"{best_result.get('sub_id')} | {best_result.get('block')}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Band Power")
        plt.legend()
        plt.xlim(start_time, stop_time)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            dat = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(os.path.join(
                save_dir,
                f"{dat}_sync_{best_channel}_{best_result.get('sub_id')}_{best_result.get('block')}_{start_time:.0f}-{stop_time:.0f}s.png"
            ))

        plt.show()

    # Final return values (GLOBAL index/time). If nothing found, return Nones
    if best_result is None:
        return None, None, None, None, None

    return (
        best_channel,
        best_result["index"],           # GLOBAL samples
        best_result["time"],            # GLOBAL seconds
        best_result,                   # dict with GLOBAL times/indices
        best_power,                    # cropped band power for best channel
    )

