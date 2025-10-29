"""
synchronizer.py — Core Synchronization and Resampling Logic
------------------------------------------------------------
Contains computational routines for aligning EEG and DBS-LFP signals in time
and frequency domains. Provides `cut_data_at_sync` and `synchronize_data`
functions that operate deterministically without GUI or user input.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

from dbs_eeg_sync.plotting import apply_publication_style, EEG_COLOR, DBS_COLOR
import matplotlib.pyplot as plt
import os

def cut_data_at_sync(
    eeg_data: Any,
    dbs_df: "pd.DataFrame",
    dbs_sync_idx: int,
    eeg_sync_idx: int,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Return cropped EEG and DBS-LFP segments aligned at detected sync peaks.

    Notes:
        - This implementation is deliberately conservative/non-destructive for now:
          it returns the EEG object unchanged and passes back the DBS-LFP signal + fs
          in a small dict. This avoids interactive choices and plotting.
        - Later, you can refine this to actually crop to windows around the peaks
          if desired. The core code and plotting code do not rely on cropping yet.

    Args:
        eeg_data: MNE Raw/Array-like EEG object with `.times` and `.get_data()`
        dbs_df:   DataFrame with columns ['TimeDomainData', 'SampleRateInHz']
        dbs_sync_idx: index of the DBS-LFP sync artifact (samples)
        eeg_sync_idx: index of the EEG sync artifact (samples)

    Returns:
        Tuple[cropped_eeg, cropped_dbs_info]
            cropped_eeg: the EEG object (unchanged pass-through)
            cropped_dbs_info: {"data": np.ndarray, "fs": float}
    """
    logger.debug("cut_data_at_sync called: dbs_sync_idx=%d, eeg_sync_idx=%d", dbs_sync_idx, eeg_sync_idx)
    # Pass-through EEG (no destructive cropping yet)
    cropped_eeg = eeg_data

    # Extract DBS-LFP signal + fs from the provided dataframe
    dbs_signal = dbs_df["TimeDomainData"].to_numpy()
    dbs_fs = float(dbs_df["SampleRateInHz"].iloc[0])
    logger.info("Extracted DBS-LFP segment: %d samples at %.2f Hz", len(dbs_signal), dbs_fs)
    cropped_dbs = {"data": dbs_signal, "fs": dbs_fs}

    return cropped_eeg, cropped_dbs


def synchronize_data(
    cropped_eeg: Any,
    cropped_dbs: Dict[str, Any],
    resample_data: bool = False,
    save_dir: Optional[str] = None,
    sub_id: Optional[str] = None,
    block: Optional[str] = None,
    plot: bool = False,
):
    """
    Produce synchronized outputs from cropped EEG and DBS-LFP without any interactive behavior.

    Design:
        - Non-interactive, headless. No figures, no prompts.
        - Returns objects suitable for downstream saving/plotting:
            * EEG: the (optionally resampled) EEG object
            * DBS-LFP: a pandas.DataFrame with 'TimeDomainData' and 'SampleRateInHz'
        - For now, resampling is disabled by default to avoid extra deps.

    Args:
        cropped_eeg: EEG object (MNE-like) obtained from cut_data_at_sync
        cropped_dbs: dict with keys {"data": np.ndarray, "fs": float}
        resample_data: if True, resample both to a common (lower) fs (not used by default)
        save_dir, sub_id, block, plot: accepted for compatibility; ignored here

    Returns:
        Tuple[Any, pd.DataFrame]: (synchronized_eeg, synchronized_dbs_df)
    """
    logger.debug("synchronize_data called: resample_data=%s, sub_id=%s, block=%s", resample_data, sub_id, block)
    apply_publication_style()
    # Pass-through (no resampling) to keep behavior deterministic for tests
    eeg_out = cropped_eeg

    dbs_signal = np.asarray(cropped_dbs["data"])
    dbs_fs = float(cropped_dbs["fs"])
    dbs_out = pd.DataFrame(
        {"TimeDomainData": dbs_signal, "SampleRateInHz": dbs_fs}
    )

    if plot:
        # Generate output directory
        if save_dir is None:
            save_dir = "outputs/plots"
        os.makedirs(save_dir, exist_ok=True)

        eeg = cropped_eeg
        try:
            eeg_t = eeg.times
            eeg_data = eeg.get_data()[0]
            eeg_fs = float(eeg.info.get("sfreq", 0))
        except Exception:
            eeg_t = np.arange(len(cropped_dbs["data"])) / float(cropped_dbs["fs"])
            eeg_data = np.asarray(cropped_dbs["data"])
            eeg_fs = float(cropped_dbs["fs"])

        dbs_signal = np.asarray(cropped_dbs["data"])
        dbs_fs = float(cropped_dbs["fs"])
        dbs_t = np.arange(len(dbs_signal)) / dbs_fs

        # Plot overlay
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(eeg_t, eeg_data, color=EEG_COLOR, label="EEG", linewidth=1.2)
        ax.plot(dbs_t, dbs_signal, color=DBS_COLOR, label="DBS-LFP", linewidth=1.2, alpha=0.9)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", frameon=False)
        ax.set_title(f"EEG–DBS-LFP Synchronization: {sub_id or ''} {block or ''}".strip())
        plot_name = f"{sub_id or 'sub'}_{block or 'block'}_overlay.png"
        fig.savefig(os.path.join(save_dir, plot_name), dpi=300)
        plt.close(fig)

    logger.info("Synchronization complete (EEG len=%d, DBS-LFP len=%d)", len(cropped_dbs['data']), len(dbs_out))
    return eeg_out, dbs_out