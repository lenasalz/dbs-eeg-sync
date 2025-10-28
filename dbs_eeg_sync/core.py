"""
core.py — EEG–DBS Synchronization Orchestration
------------------------------------------------
Defines the top-level synchronization function `sync_run`, which loads EEG
and DBS data, detects stimulation artifacts, aligns signals in time, and
returns a structured `SyncResult`. Designed to be non-interactive and reusable
both as a library function and through the CLI.
"""

# dbs_eeg_sync/core.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Dict

# Import existing modules from the legacy 'source' package.
from source.data_loader import (
    load_eeg_data, dbs_artifact_settings, open_json_file,
    select_recording, read_time_domain_data
)
from source.sync_artefact_finder import (
    detect_dbs_sync_artifact, detect_eeg_sync_artifact
)
from source.synchronizer import (
    cut_data_at_sync, synchronize_data
)

@dataclass(frozen=True)
class SyncResult:
    sub_id: str
    block: str
    eeg_file: Path
    dbs_file: Path

    # artifact picks (indices and seconds)
    eeg_sync_idx: int
    eeg_sync_s: float
    dbs_sync_idx: int
    dbs_sync_s: float

    # sample rates
    eeg_fs: float
    dbs_fs: float

    # synchronized/cropped objects to be saved or further processed
    synchronized_eeg: Any     # keep Any to avoid hard coupling to MNE/pandas types here
    synchronized_dbs: Any
    cropped_eeg: Any
    cropped_dbs: Any

    # extras that help provenance/reproducibility
    channel: Optional[str]
    artifact_kind: Optional[str]
    artifact_magnitude: Optional[float]
    time_range: Optional[Tuple[float, float]]
    metadata: dict


class SyncError(Exception):
    """Raised when synchronization cannot be completed."""

def _validate_paths(eeg_file: Path, dbs_file: Path):
    if not eeg_file.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")
    if not dbs_file.exists():
        raise FileNotFoundError(f"DBS file not found: {dbs_file}")

def _validate_time_range(time_range, eeg_duration: float):
    if time_range is None:
        return
    a, b = time_range
    if not (0.0 <= a < b <= eeg_duration):
        raise ValueError(
            f"Invalid time_range={time_range}. Must satisfy 0 <= start < end <= {eeg_duration:.3f}"
        )

def sync_run(
    *,
    sub_id: str,
    block: str,
    eeg_file: Path,
    dbs_file: Path,
    time_range: tuple[float, float] | None = None,
    block_index: int | None = 0,                 # default 0 to avoid prompt
    artifact_band: tuple[float, float] | None = (120.0, 130.0),  # avoid prompt
) -> SyncResult:
    """
    Run end-to-end synchronization without any I/O side effects (no prints, no GUI, no plots, no saving).
    Returns a SyncResult with indices, timing info, data objects, and metadata.
    Raises SyncError/ValueError/FileNotFoundError on failures.
    """
    eeg_file = Path(eeg_file)
    dbs_file = Path(dbs_file)
    _validate_paths(eeg_file, dbs_file)

    # Load EEG
    eeg_data, eeg_fs = load_eeg_data(str(eeg_file))
    eeg_duration = float(eeg_data.times[-1])
    _validate_time_range(time_range, eeg_duration)

    # Load DBS JSON and extract time-domain block
    json_data = open_json_file(str(dbs_file))
    block_num = select_recording(json_data, block_index)
    dbs_df = read_time_domain_data(json_data, block_num)
    dbs_signal = dbs_df["TimeDomainData"].to_numpy()
    dbs_fs = float(dbs_df["SampleRateInHz"].iloc[0])

    # Artifact frequency range (from your helper)
    dbs_freq_min, dbs_freq_max, _, _ = dbs_artifact_settings(
        freq_min=artifact_band[0] if artifact_band else None,
        freq_max=artifact_band[1] if artifact_band else None,
        interactive=False
    )

    # Detect EEG artifact (no plotting in core)
    channel, eeg_sync_idx, eeg_sync_s, result, smoothed_power = detect_eeg_sync_artifact(
        eeg_data,
        freq_low=dbs_freq_min,
        freq_high=dbs_freq_max,
        time_range=time_range,
        plot=False,            # no plots in core
        save_dir=None,         # no saving in core
        sub_id=sub_id,
        block=block,
    )
    if channel is None or eeg_sync_idx is None:
        raise SyncError("EEG sync detection failed (no channel/index returned).")

    # Detect DBS artifact (no plots in core)
    dbs_sync_idx, dbs_sync_s = detect_dbs_sync_artifact(
        dbs_signal,
        dbs_fs,
        save_dir=None,
        sub_id=sub_id,
        block=block,
        plot=False,
    )

    # Crop and synchronize (no plots in core)
    cropped_eeg, cropped_dbs = cut_data_at_sync(
        eeg_data, dbs_df, dbs_sync_idx, eeg_sync_idx
    )
    synchronized_eeg, synchronized_dbs = synchronize_data(
        cropped_eeg, cropped_dbs, resample_data=False,
        save_dir=None, sub_id=sub_id, block=block, plot=False
    )

    # Prepare metadata (for provenance; callers may persist this)
    meta: Dict = {
        "artifact_band_hz": [dbs_freq_min, dbs_freq_max],
        "eeg_duration_s": eeg_duration,
        "time_range": time_range,
        "detector": {
            "channel": channel,
            "kind": result.get("type") if isinstance(result, dict) else None,
            "magnitude": result.get("magnitude") if isinstance(result, dict) else None,
        },
        "fs": {"eeg": eeg_fs, "dbs": dbs_fs},
        "block_num": int(block_num),
    }

    return SyncResult(
        sub_id=sub_id,
        block=block,
        eeg_file=eeg_file,
        dbs_file=dbs_file,
        eeg_sync_idx=int(eeg_sync_idx),
        eeg_sync_s=float(eeg_sync_s),
        dbs_sync_idx=int(dbs_sync_idx),
        dbs_sync_s=float(dbs_sync_s),
        eeg_fs=float(eeg_fs),
        dbs_fs=float(dbs_fs),
        synchronized_eeg=synchronized_eeg,
        synchronized_dbs=synchronized_dbs,
        cropped_eeg=cropped_eeg,
        cropped_dbs=cropped_dbs,
        channel=channel,
        artifact_kind=meta["detector"]["kind"],
        artifact_magnitude=meta["detector"]["magnitude"],
        time_range=time_range,
        metadata=meta,
    )