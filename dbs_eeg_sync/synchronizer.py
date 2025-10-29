# dbs_eeg_sync/synchronizer.py

from __future__ import annotations
from typing import Tuple, Optional
import logging
from pathlib import Path
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.signal import resample as sp_resample

# mne is an optional heavy dep but required for Raw handling here
import mne

from dbs_eeg_sync.plotting import apply_publication_style, EEG_COLOR, DBS_COLOR

logger = logging.getLogger(__name__)


def cut_data_at_sync(
    eeg_raw: mne.io.BaseRaw,
    dbs_df: pd.DataFrame,
    dbs_sync_idx: int,
    eeg_sync_idx: int,
) -> Tuple[mne.io.BaseRaw, pd.DataFrame]:
    """
    Crop EEG and DBS from their respective sync indices onward.

    Args:
        eeg_raw: mne Raw (or BaseRaw) object.
        dbs_df:  Pandas DataFrame with columns ["TimeDomainData", "SampleRateInHz"].
                 Assumed single, continuous DBS trace (1D).
        dbs_sync_idx: index (in DBS samples) at which the DBS sync artifact occurs.
        eeg_sync_idx: index (in EEG samples) at which the EEG sync artifact occurs.

    Returns:
        (cropped_eeg_raw, cropped_dbs_df)
    """
    # --- Validate inputs
    if not isinstance(eeg_raw, mne.io.BaseRaw):
        raise TypeError("eeg_raw must be an mne.io.BaseRaw (e.g., Raw).")
    if not isinstance(dbs_df, pd.DataFrame):
        raise TypeError("dbs_df must be a pandas.DataFrame.")
    for col in ("TimeDomainData", "SampleRateInHz"):
        if col not in dbs_df.columns:
            raise ValueError(f"dbs_df must contain column '{col}'.")

    eeg_fs = float(eeg_raw.info["sfreq"])
    dbs_fs = float(dbs_df["SampleRateInHz"].iloc[0])

    # --- Bounds checks
    n_eeg = eeg_raw.n_times  # samples
    n_dbs = len(dbs_df["TimeDomainData"])
    if not (0 <= eeg_sync_idx <= n_eeg):
        raise ValueError(f"eeg_sync_idx {eeg_sync_idx} out of range [0, {n_eeg}].")
    if not (0 <= dbs_sync_idx <= n_dbs):
        raise ValueError(f"dbs_sync_idx {dbs_sync_idx} out of range [0, {n_dbs}].")

    # --- Crop EEG from eeg_sync_idx onward (convert samples -> seconds)
    tmin_sec = eeg_sync_idx / eeg_fs
    cropped_eeg = eeg_raw.copy().crop(tmin=tmin_sec)

    # --- Crop DBS from dbs_sync_idx onward
    cropped_dbs = dbs_df.iloc[dbs_sync_idx:].reset_index(drop=True).copy()

    logger.info(
        "Cropped at sync: EEG t>=%.3fs (from %d/%d samples), DBS i>=%d/%d samples.",
        tmin_sec, eeg_sync_idx, n_eeg, dbs_sync_idx, n_dbs
    )
    return cropped_eeg, cropped_dbs


def synchronize_data(
    cropped_eeg: mne.io.BaseRaw,
    cropped_dbs: pd.DataFrame,
    resample_data: bool = True,
    save_dir: Optional[str] = "outputs/plots",
    sub_id: Optional[str] = None,
    block: Optional[str] = None,
) -> Tuple[mne.io.BaseRaw, pd.DataFrame]:
    """
    Optionally resample EEG & DBS to a common (lower) sampling rate (deterministic, non-interactive).
    Also saves a clean overlay plot if save_dir is provided.

    Args:
        cropped_eeg: mne Raw after cropping at sync.
        cropped_dbs: DataFrame with ["TimeDomainData", "SampleRateInHz"] after cropping.
        resample_data: If True, resample both to min(eeg_fs, dbs_fs).
        save_dir: If not None, save overlay figure into this directory.
        sub_id, block: Used for figure naming.

    Returns:
        (eeg_out, dbs_out)
            eeg_out: possibly resampled Raw
            dbs_out: DataFrame with resampled "TimeDomainData" and updated "SampleRateInHz"
    """
    # --- Validate
    if not isinstance(cropped_eeg, mne.io.BaseRaw):
        raise TypeError("cropped_eeg must be an mne.io.BaseRaw.")
    if not isinstance(cropped_dbs, pd.DataFrame):
        raise TypeError("cropped_dbs must be a pandas.DataFrame.")
    for col in ("TimeDomainData", "SampleRateInHz"):
        if col not in cropped_dbs.columns:
            raise ValueError(f"cropped_dbs must contain column '{col}'.")

    eeg_fs = float(cropped_eeg.info["sfreq"])
    dbs_fs = float(cropped_dbs["SampleRateInHz"].iloc[0])
    dbs_signal = np.asarray(cropped_dbs["TimeDomainData"].values, dtype=float)

    target_fs = min(eeg_fs, dbs_fs)

    # --- Resample (if requested)
    if resample_data:
        eeg_out = cropped_eeg.copy()
        if eeg_fs != target_fs:
            eeg_out.resample(sfreq=target_fs)

        if dbs_fs != target_fs:
            new_len = int(round(len(dbs_signal) * (target_fs / dbs_fs)))
            dbs_signal_rs = sp_resample(dbs_signal, new_len)
            dbs_out = pd.DataFrame(
                {"TimeDomainData": dbs_signal_rs, "SampleRateInHz": target_fs}
            )
        else:
            dbs_out = pd.DataFrame(
                {"TimeDomainData": dbs_signal, "SampleRateInHz": dbs_fs}
            )
    else:
        eeg_out = cropped_eeg
        dbs_out = pd.DataFrame(
            {"TimeDomainData": dbs_signal, "SampleRateInHz": dbs_fs}
        )

    logger.info(
        "Synchronization (resample=%s): EEG fs=%.3f→%.3f, DBS fs=%.3f→%.3f.",
        resample_data,
        eeg_fs, float(eeg_out.info['sfreq']),
        dbs_fs, float(dbs_out['SampleRateInHz'].iloc[0]),
    )

    # --- Optional figure
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        apply_publication_style()

        # Time vectors
        eeg_data = eeg_out.get_data()[0]  # first channel for a simple overlay
        fs_eeg = float(eeg_out.info["sfreq"])
        t_eeg = np.arange(eeg_data.size) / fs_eeg

        dbs_sig = dbs_out["TimeDomainData"].to_numpy()
        fs_dbs = float(dbs_out["SampleRateInHz"].iloc[0])
        t_dbs = np.arange(dbs_sig.size) / fs_dbs

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t_eeg, eeg_data, label="EEG", color=EEG_COLOR, linewidth=1.2)
        ax.plot(t_dbs, dbs_sig, label="DBS-LFP", color=DBS_COLOR, linewidth=1.2, alpha=0.9)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", frameon=False)
        ax.set_title(f"EEG–DBS Synchronization: {sub_id or ''} {block or ''}".strip())

        fname = f"{sub_id or 'sub'}_{block or 'block'}_overlay.png"
        out_path = Path(save_dir) / fname
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved overlay plot: %s", out_path)

    return eeg_out, dbs_out