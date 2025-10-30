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
import matplotlib as mpl

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
    apply_publication_style()

    t = np.arange(len(dbs_signal)) / float(dbs_fs)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(t, dbs_signal, label="LFP", color=DBS_COLOR)
    ax.axvline(peak_idx / float(dbs_fs), linestyle="--", color=DBS_COLOR, alpha=0.8,
               label=f"artifact @ {peak_idx/float(dbs_fs):.2f}s")
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
    apply_publication_style()

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(time_s, power, label=f"Power {channel or ''}", color=EEG_COLOR)
    if event_time is not None:
        ax.axvline(event_time, color=ACCENT_COLOR, linestyle="--",
                   label=f"event @ {event_time:.2f}s")
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
    apply_publication_style()

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(eeg_t, eeg_y, label="EEG (synced)", color=EEG_COLOR)
    ax.plot(dbs_t, dbs_y, label="LFP (synced)", color=DBS_COLOR, alpha=0.85)
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


# Consistent brand-ish colors
EEG_COLOR = "#CD5F66"   # red-ish
DBS_COLOR = "#5794A0"   # turquoise-ish
ACCENT_COLOR = "#2ca02c"  # green (if ever needed)

def apply_publication_style() -> None:
    """Set a clean, publication-ready Matplotlib style globally."""
    mpl.rcParams.update({
        # sizing
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.autolayout": True,  # like tight_layout
        "figure.figsize": (10, 3),
        # fonts
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        # lines
        "lines.linewidth": 1.0,
        "lines.antialiased": True,
        # save
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })