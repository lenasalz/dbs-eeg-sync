"""
plotting.py — Visualization Utilities for EEG–DBS Synchronization
------------------------------------------------------------------
Contains non-interactive Matplotlib plotting functions for synchronized EEG
and DBS data. All functions support headless operation (no GUI) and save
figures to disk for reproducible analysis and publication-quality figures.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

def _ensure_dir(p: Optional[Path | str]) -> Optional[Path]:
    if p is None:
        return None
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def plot_dbs_artifact(
    dbs_signal,
    dbs_fs: float,
    peak_idx: int,
    *,
    outdir: Optional[Path | str] = None,
    sub_id: Optional[str] = None,
    block: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot the DBS signal with the detected artifact. Saves a PNG if outdir is provided; shows a window only if show=True."""
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.arange(len(dbs_signal)) / float(dbs_fs)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(t, dbs_signal, label="DBS")
    ax.axvline(peak_idx / float(dbs_fs), linestyle="--", label=f"artifact @ {peak_idx/float(dbs_fs):.2f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()

    if outdir:
        outdir = _ensure_dir(outdir)
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(Path(outdir) / f"{dat}_syncDBS_{sub_id}_{block}.png")

    if show:
        plt.show()
    plt.close(fig)

def plot_eeg_power(
    time_s,
    power,
    *,
    event_time: Optional[float] = None,
    channel: Optional[str] = None,
    outdir: Optional[Path | str] = None,
    sub_id: Optional[str] = None,
    block: Optional[str] = None,
    show: bool = False,
    filename_prefix: str = "sync",
) -> None:
    """Plot EEG band power time-course with optional vertical event line. Saves PNG if outdir; show only if show=True."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(time_s, power, label=f"Power {channel or ''}")
    if event_time is not None:
        ax.axvline(event_time, color="red", linestyle="--", label=f"event @ {event_time:.2f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Band Power")
    ax.legend()
    fig.tight_layout()

    if outdir:
        outdir = _ensure_dir(outdir)
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        ch = (channel or "NA").replace("/", "_")
        fig.savefig(Path(outdir) / f"{dat}_{filename_prefix}_{ch}_{sub_id}_{block}.png")

    if show:
        plt.show()
    plt.close(fig)

def plot_eeg_dbs_overlay(
    eeg_t,
    eeg_y,
    dbs_t,
    dbs_y,
    *,
    outdir: Optional[Path | str] = None,
    sub_id: Optional[str] = None,
    block: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot synchronized EEG and DBS signals overlaid in time. Saves PNG if outdir; show only if show=True."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(eeg_t, eeg_y, label="EEG (synced)")
    ax.plot(dbs_t, dbs_y, label="DBS (synced)", alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()

    if outdir:
        outdir = _ensure_dir(outdir)
        dat = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(Path(outdir) / f"{dat}_eeg_dbs_overlay_{sub_id}_{block}.png")

    if show:
        plt.show()
    plt.close(fig)