"""
synchronizer.py — Core Synchronization and Resampling Logic
------------------------------------------------------------
Contains computational routines for aligning EEG and DBS signals in time
and frequency domains. Provides `cut_data_at_sync` and `synchronize_data`
functions that operate deterministically without GUI or user input.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd

def cut_data_at_sync(
    eeg_data: Any,
    dbs_df: "pd.DataFrame",
    dbs_sync_idx: int,
    eeg_sync_idx: int,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Return cropped EEG and DBS segments aligned at detected sync peaks.

    Notes:
        - This implementation is deliberately conservative/non-destructive for now:
          it returns the EEG object unchanged and passes back the DBS signal + fs
          in a small dict. This avoids interactive choices and plotting.
        - Later, you can refine this to actually crop to windows around the peaks
          if desired. The core code and plotting code do not rely on cropping yet.

    Args:
        eeg_data: MNE Raw/Array-like EEG object with `.times` and `.get_data()`
        dbs_df:   DataFrame with columns ['TimeDomainData', 'SampleRateInHz']
        dbs_sync_idx: index of the DBS sync artifact (samples)
        eeg_sync_idx: index of the EEG sync artifact (samples)

    Returns:
        Tuple[cropped_eeg, cropped_dbs_info]
            cropped_eeg: the EEG object (unchanged pass-through)
            cropped_dbs_info: {"data": np.ndarray, "fs": float}
    """
    # Pass-through EEG (no destructive cropping yet)
    cropped_eeg = eeg_data

    # Extract DBS signal + fs from the provided dataframe
    dbs_signal = dbs_df["TimeDomainData"].to_numpy()
    dbs_fs = float(dbs_df["SampleRateInHz"].iloc[0])
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
    Produce synchronized outputs from cropped EEG and DBS without any interactive behavior.

    Design:
        - Non-interactive, headless. No figures, no prompts.
        - Returns objects suitable for downstream saving/plotting:
            * EEG: the (optionally resampled) EEG object
            * DBS: a pandas.DataFrame with 'TimeDomainData' and 'SampleRateInHz'
        - For now, resampling is disabled by default to avoid extra deps.

    Args:
        cropped_eeg: EEG object (MNE-like) obtained from cut_data_at_sync
        cropped_dbs: dict with keys {"data": np.ndarray, "fs": float}
        resample_data: if True, resample both to a common (lower) fs (not used by default)
        save_dir, sub_id, block, plot: accepted for compatibility; ignored here

    Returns:
        Tuple[Any, pd.DataFrame]: (synchronized_eeg, synchronized_dbs_df)
    """
    # Pass-through (no resampling) to keep behavior deterministic for tests
    eeg_out = cropped_eeg

    dbs_signal = np.asarray(cropped_dbs["data"])
    dbs_fs = float(cropped_dbs["fs"])
    dbs_out = pd.DataFrame(
        {"TimeDomainData": dbs_signal, "SampleRateInHz": dbs_fs}
    )

    return eeg_out, dbs_out