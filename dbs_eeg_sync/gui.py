from __future__ import annotations

import logging
from typing import Tuple

from dbs_eeg_sync.power_calculater import compute_samplewise_eeg_power
from dbs_eeg_sync.plotting import apply_publication_style, EEG_COLOR, DBS_COLOR


logger = logging.getLogger(__name__)


def _to_1d_signal(x, *, prefer_axis="channels_first"):
    """
    Convert an arbitrary EEG/DBS container into a 1-D numpy array (time series).

    - Supports MNE Raw-like objects via .get_data()
    - Supports numpy arrays (1-D or 2-D). If 2-D, reduces across channels by mean.
    - Supports sequences/lists of equal-length arrays by stacking then mean across rows.
    """
    import numpy as _np

    # MNE Raw-like
    if hasattr(x, "get_data"):
        try:
            arr = x.get_data()  # (n_channels, n_times)
        except TypeError:
            # Some MNE objects require picks=None explicitly
            arr = x.get_data(picks=None)
    else:
        # Try array conversion first without forcing ragged object arrays
        try:
            arr = _np.asarray(x)
        except Exception:
            # Last resort: try to stack if it's a list/tuple of arrays
            if isinstance(x, (list, tuple)):
                arr = _np.vstack([_np.asarray(xx) for xx in x])
            else:
                raise

        # If we got an object dtype (ragged), attempt to stack rows if possible
        if arr.dtype == object:
            if isinstance(x, (list, tuple)):
                try:
                    arr = _np.vstack([_np.asarray(xx) for xx in x])
                except Exception as exc:
                    raise ValueError(
                        "Cannot form a regular 2-D array from a ragged sequence."
                    ) from exc

    # Now arr should be numeric
    arr = _np.asarray(arr, dtype=float)

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        # Determine which axis is channels and which is time
        n0, n1 = arr.shape
        # Heuristic: channels-first is the usual (n_channels, n_times)
        if prefer_axis == "channels_first":
            # If the first dim is small-ish (like channels) and second is large (time), keep it
            pass
        # Reduce across channels to a representative 1-D series (mean)
        if n0 <= n1:
            # (n_channels, n_times)
            return _np.nanmean(arr, axis=0)
        else:
            # (n_times, n_channels) — rare, but handle
            return _np.nanmean(arr, axis=1)

    # More than 2-D: flatten last axis as time and reduce others
    # Collapse everything but the last axis by mean
    if arr.ndim > 2:
        reshaped = arr.reshape(-1, arr.shape[-1])  # (prod(other_dims), n_times)
        return _np.nanmean(reshaped, axis=0)

    raise ValueError("Unsupported input shape for signal conversion.")


def _import_qt():
    """
    Try PyQt6 first, then PyQt5. Return (QtCore, QtWidgets, QtGui, is_qt6).
    Raises ImportError if neither is available.
    """
    try:
        from PyQt6 import QtCore, QtWidgets, QtGui  # type: ignore
        return QtCore, QtWidgets, QtGui, True
    except Exception:
        from PyQt5 import QtCore, QtWidgets, QtGui  # type: ignore
        return QtCore, QtWidgets, QtGui, False


def _require_qt_backend():
    """
    Ensure matplotlib is using a Qt backend. We don't force-switch here
    (to avoid surprising users), we just validate and error out cleanly.
    """
    import matplotlib
    backend = str(matplotlib.get_backend()).lower()
    if "qt" not in backend:
        raise RuntimeError(
            f"GUI requested but matplotlib backend is '{matplotlib.get_backend()}'. "
            "Use a Qt backend (QtAgg) and make sure a display is available."
        )


class _ManualSyncWidget(object):
    """
    Helper that builds the widget once we know Qt is available.
    Split into a plain class so we can keep all imports lazy.
    """
    def __init__(self, QtCore, QtWidgets, QtGui, np, FigureCanvas, eeg_data, eeg_fs, dbs_data, dbs_fs, title, freq_low=None, freq_high=None, channel=None):
        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.QtGui = QtGui
        self.np = np
        self.FigureCanvas = FigureCanvas

        # EEG: prefer band-power of a single channel if Raw-like
        self.eeg_fs = float(eeg_fs)
        self.dbs_fs = float(dbs_fs)

        eeg_signal = None
        eeg_title_suffix = ""
        try:
            # Raw-like with channel names
            if hasattr(eeg_data, "get_data") and hasattr(eeg_data, "ch_names"):
                ch_names = list(getattr(eeg_data, "ch_names", []))
                # Choose channel
                preferred = ['T8','T7','Cz','Pz','Oz','Fz','O1','O2']
                ch = channel if (channel and channel in ch_names) else next((c for c in preferred if c in ch_names), ch_names[0] if ch_names else None)
                # Band defaults
                f_lo = 120.0 if freq_low is None else float(freq_low)
                f_hi = 130.0 if freq_high is None else float(freq_high)
                # Compute per-sample band power for the chosen channel
                power, t_power = compute_samplewise_eeg_power(
                    eeg_data, freq_low=f_lo, freq_high=f_hi, channel=ch,
                )
                eeg_signal = np.asarray(power, dtype=float).ravel()
                self.t_eeg = np.asarray(t_power, dtype=float).ravel()
                eeg_title_suffix = f" (power, {ch}, {f_lo:.1f}-{f_hi:.1f} Hz)"
                self.eeg_channel = ch
            else:
                # Fallback: reduce to 1D time series
                eeg_signal = _to_1d_signal(eeg_data)
                self.t_eeg = np.arange(np.asarray(eeg_signal).size, dtype=float) / self.eeg_fs
                self.eeg_channel = None
        except Exception:
            # Robust fallback
            eeg_signal = _to_1d_signal(eeg_data)
            self.t_eeg = np.arange(np.asarray(eeg_signal).size, dtype=float) / self.eeg_fs
            self.eeg_channel = None

        self.eeg = np.asarray(eeg_signal, dtype=float)

        # DBS signal always as simple 1D
        self.dbs = _to_1d_signal(dbs_data)
        self.t_dbs = np.arange(self.dbs.size, dtype=float) / self.dbs_fs

        self.result = None  # (eeg_idx, eeg_t, dbs_idx, dbs_t)

        self.widget = QtWidgets.QWidget()
        self.widget.setWindowTitle(title or "dbs-eeg-sync | Manual sync")

        # ----- Layouts
        main = QtWidgets.QVBoxLayout(self.widget)

        # Matplotlib canvases (QtAgg works for both PyQt5/6 via backend_qtagg)
        import matplotlib.pyplot as plt
        apply_publication_style()
        self.fig_eeg, self.ax_eeg = plt.subplots(1, 1, figsize=(10, 3))
        self.fig_dbs, self.ax_dbs = plt.subplots(1, 1, figsize=(10, 3))

        self.canvas_eeg = self.FigureCanvas(self.fig_eeg)
        self.canvas_dbs = self.FigureCanvas(self.fig_dbs)

        # EEG plot
        eeg_label = "EEG power" if "power" in eeg_title_suffix else "EEG"
        self.ax_eeg.plot(self.t_eeg, self.eeg, linewidth=1.2, color=EEG_COLOR, label=eeg_label)
        self.ax_eeg.set_title("EEG — choose one index" + eeg_title_suffix)
        self.ax_eeg.set_xlabel("Time [s]")
        self.ax_eeg.set_ylabel("Power" if "power" in eeg_title_suffix else "Amplitude")
        self.line_eeg = self.ax_eeg.axvline(0.0, linestyle="--", color="0.25", linewidth=1.0)
        self.ax_eeg.legend(loc="upper right", frameon=False)
        self.fig_eeg.tight_layout()

        # DBS plot
        self.ax_dbs.plot(self.t_dbs, self.dbs, linewidth=1.2, color=DBS_COLOR, label="DBS-LFP")
        self.ax_dbs.set_title("DBS-LFP — choose one index")
        self.ax_dbs.set_xlabel("Time [s]")
        self.ax_dbs.set_ylabel("Amplitude")
        self.line_dbs = self.ax_dbs.axvline(0.0, linestyle="--", color="0.25", linewidth=1.0)
        self.ax_dbs.legend(loc="upper right", frameon=False)
        self.fig_dbs.tight_layout()

        # Sliders + labels
        self.label_eeg = QtWidgets.QLabel("EEG: idx=0  t=0.000s")
        self.slider_eeg = QtWidgets.QSlider(self.QtCore.Qt.Orientation.Horizontal)
        self.slider_eeg.setMinimum(0)
        self.slider_eeg.setMaximum(max(0, self.eeg.size - 1))
        self.slider_eeg.setValue(0)
        self.slider_eeg.valueChanged.connect(self._update_eeg)

        self.label_dbs = QtWidgets.QLabel("DBS-LFP: idx=0  t=0.000s")
        self.slider_dbs = QtWidgets.QSlider(self.QtCore.Qt.Orientation.Horizontal)
        self.slider_dbs.setMinimum(0)
        self.slider_dbs.setMaximum(max(0, self.dbs.size - 1))
        self.slider_dbs.setValue(0)
        self.slider_dbs.valueChanged.connect(self._update_dbs)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_ok = QtWidgets.QPushButton("Confirm selection")
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_ok.clicked.connect(self._on_ok)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_ok)

        # Assemble
        main.addWidget(self.canvas_eeg)
        main.addWidget(self.label_eeg)
        main.addWidget(self.slider_eeg)

        main.addSpacing(8)

        main.addWidget(self.canvas_dbs)
        main.addWidget(self.label_dbs)
        main.addWidget(self.slider_dbs)

        main.addSpacing(8)
        main.addLayout(btn_row)

        # Initial draw
        self._update_eeg(0)
        self._update_dbs(0)

    # ----- Slots
    def _update_eeg(self, idx: int):
        idx = int(idx)
        idx = max(0, min(idx, self.eeg.size - 1))
        t = idx / self.eeg_fs
        self.label_eeg.setText(f"EEG: idx={idx}  t={t:.3f}s")
        self.line_eeg.set_xdata([t, t])
        self.canvas_eeg.draw_idle()

    def _update_dbs(self, idx: int):
        idx = int(idx)
        idx = max(0, min(idx, self.dbs.size - 1))
        t = idx / self.dbs_fs
        self.label_dbs.setText(f"DBS-LFP: idx={idx}  t={t:.3f}s")
        self.line_dbs.set_xdata([t, t])
        self.canvas_dbs.draw_idle()

    def _on_cancel(self):
        logger.info("Manual GUI canceled by user.")
        self.result = None
        self.widget.close()

    def _on_ok(self):
        eeg_idx = int(self.slider_eeg.value())
        dbs_idx = int(self.slider_dbs.value())
        eeg_t = float(eeg_idx / self.eeg_fs)
        dbs_t = float(dbs_idx / self.dbs_fs)
        self.result = (eeg_idx, eeg_t, dbs_idx, dbs_t)
        logger.info(
            "Manual picks — EEG: idx=%d, t=%.3fs | DBS-LFP: idx=%d, t=%.3fs",
            eeg_idx, eeg_t, dbs_idx, dbs_t
        )
        self.widget.close()


def manual_select_sync(
    eeg_data,  # Raw or array-like
    eeg_fs: float,
    dbs_data,  # 1D array-like
    dbs_fs: float,
    title: str,
    *,
    freq_low: float | None = None,
    freq_high: float | None = None,
    channel: str | None = None,
) -> Tuple[int, float, int, float]:
    """
    Slider-based manual selector using Qt + matplotlib (QtAgg).
    Returns (eeg_idx, eeg_t, dbs_idx, dbs_t).
    """
    import importlib

    # Validate backend first (clear message if not Qt)
    _require_qt_backend()

    # Lazy imports
    np = importlib.import_module("numpy")

    # Qt (PyQt6 -> PyQt5)
    try:
        QtCore, QtWidgets, QtGui, is_qt6 = _import_qt()
    except Exception as exc:
        logger.debug("Qt import failed", exc_info=True)
        raise RuntimeError(
            "GUI mode requested but neither PyQt6 nor PyQt5 is installed."
        ) from exc

    # Matplotlib canvas (qtagg works for Qt5/6)
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except Exception:
        # Older Matplotlib may not have backend_qtagg; fall back to qt5agg
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # type: ignore

    # Create / reuse QApplication
    app = QtWidgets.QApplication.instance()
    owns_app = False
    if app is None:
        app = QtWidgets.QApplication(["dbs-eeg-sync"])
        owns_app = True

    # Build widget
    gui = _ManualSyncWidget(
        QtCore, QtWidgets, QtGui, np, FigureCanvas,
        eeg_data, eeg_fs, dbs_data, dbs_fs, title,
        freq_low=freq_low, freq_high=freq_high, channel=channel,
    )
    gui.widget.resize(1100, 800)
    gui.widget.show()

    # Exec event loop if we created the app, else run a modal loop
    if owns_app:
        app.exec()
    else:
        # Process events until window closes
        while gui.widget.isVisible():
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents if hasattr(QtCore.QEventLoop, "ProcessEventsFlag") else QtCore.QEventLoop.AllEvents)

    if gui.result is None:
        raise RuntimeError("Manual selection cancelled.")

    return gui.result


__all__ = ["manual_select_sync"]