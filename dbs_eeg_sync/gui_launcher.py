"""
gui_launcher.py — Double-click / no-command-line entry point for clinicians.

This module provides a small graphical front-end so that users who are not
comfortable with the command line can:

  1. Pick an EEG file and a DBS JSON file with normal "Open file" dialogs.
  2. Fill in a subject ID, block label, and (optionally) a time range.
  3. Press "Run synchronization".

It then calls the same :func:`dbs_eeg_sync.core.sync_run` pipeline that the
command-line interface uses, with the manual-selection GUI enabled, and saves
metadata + plots to a chosen output folder.

The heavy imports (Qt, matplotlib, the core pipeline) are deferred until the
window is actually opened so that ``python -c "import dbs_eeg_sync"`` stays
cheap and so that a missing optional dependency produces a clear message
instead of an import crash.

Run it with either::

    dbs-eeg-sync-gui          # installed console script
    python -m dbs_eeg_sync    # module entry point
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


_QT_INSTALL_HINT = (
    "The graphical interface needs PyQt6, which is an optional dependency.\n\n"
    "Install it (inside your environment) with:\n\n"
    '    pip install "dbs-eeg-sync[gui]"\n\n'
    "or:\n\n"
    "    pip install PyQt6\n"
)


def _import_qt():
    """Return (QtWidgets, QtCore) for PyQt6 or PyQt5, or raise a clear error."""
    try:
        from PyQt6 import QtCore, QtWidgets  # type: ignore

        return QtWidgets, QtCore
    except Exception:  # pragma: no cover - exercised only without PyQt6
        try:
            from PyQt5 import QtCore, QtWidgets  # type: ignore

            return QtWidgets, QtCore
        except Exception as exc:
            raise RuntimeError(_QT_INSTALL_HINT) from exc


class _LauncherWindow:
    """A minimal form: file pickers + a few text fields + a Run button."""

    def __init__(self, QtWidgets, QtCore):
        self.QtWidgets = QtWidgets
        self.QtCore = QtCore

        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("dbs-eeg-sync — EEG/DBS synchronization")
        self.window.resize(640, 0)

        form = QtWidgets.QFormLayout()

        # --- EEG file row
        self.eeg_edit = QtWidgets.QLineEdit()
        eeg_btn = QtWidgets.QPushButton("Browse…")
        eeg_btn.clicked.connect(self._pick_eeg)
        form.addRow("EEG file:", self._row(self.eeg_edit, eeg_btn))

        # --- DBS file row
        self.dbs_edit = QtWidgets.QLineEdit()
        dbs_btn = QtWidgets.QPushButton("Browse…")
        dbs_btn.clicked.connect(self._pick_dbs)
        form.addRow("DBS JSON file:", self._row(self.dbs_edit, dbs_btn))

        # --- Output folder row
        self.out_edit = QtWidgets.QLineEdit("outputs")
        out_btn = QtWidgets.QPushButton("Browse…")
        out_btn.clicked.connect(self._pick_outdir)
        form.addRow("Output folder:", self._row(self.out_edit, out_btn))

        # --- Subject / block / time range
        self.sub_edit = QtWidgets.QLineEdit("S01")
        form.addRow("Subject ID:", self.sub_edit)

        self.block_edit = QtWidgets.QLineEdit("B1")
        form.addRow("Block / session:", self.block_edit)

        self.range_edit = QtWidgets.QLineEdit()
        self.range_edit.setPlaceholderText("optional, e.g. 10,60 (seconds)")
        form.addRow("Time range:", self.range_edit)

        self.manual_check = QtWidgets.QCheckBox(
            "Pick sync point by hand (opens the slider window)"
        )
        self.manual_check.setChecked(True)
        form.addRow("", self.manual_check)

        self.plots_check = QtWidgets.QCheckBox("Save plots")
        self.plots_check.setChecked(True)
        form.addRow("", self.plots_check)

        # --- Status line + Run button
        self.status = QtWidgets.QLabel("Choose your files, then press Run.")
        self.status.setWordWrap(True)

        run_btn = QtWidgets.QPushButton("Run synchronization")
        run_btn.setStyleSheet(
            "QPushButton { font-size: 12pt; padding: 10px; font-weight: bold; "
            "background-color: #4CAF50; color: white; } "
            "QPushButton:hover { background-color: #45a049; }"
        )
        run_btn.clicked.connect(self._run)

        layout = QtWidgets.QVBoxLayout(self.window)
        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addWidget(run_btn)

    def _row(self, line_edit, button):
        w = self.QtWidgets.QWidget()
        h = self.QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(line_edit)
        h.addWidget(button)
        return w

    # ----- file dialogs
    def _pick_eeg(self):
        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Select EEG file",
            "",
            "EEG files (*.set *.edf *.fif *.vhdr);;All files (*)",
        )
        if path:
            self.eeg_edit.setText(path)

    def _pick_dbs(self):
        path, _ = self.QtWidgets.QFileDialog.getOpenFileName(
            self.window,
            "Select DBS JSON file",
            "",
            "DBS recordings (*.json);;All files (*)",
        )
        if path:
            self.dbs_edit.setText(path)

    def _pick_outdir(self):
        path = self.QtWidgets.QFileDialog.getExistingDirectory(
            self.window, "Select output folder"
        )
        if path:
            self.out_edit.setText(path)

    def _error(self, message: str) -> None:
        self.QtWidgets.QMessageBox.critical(self.window, "dbs-eeg-sync", message)

    def _info(self, message: str) -> None:
        self.QtWidgets.QMessageBox.information(self.window, "dbs-eeg-sync", message)

    def _parse_time_range(self):
        text = self.range_edit.text().strip()
        if not text:
            return None
        parts = text.replace(" ", "").split(",")
        if len(parts) != 2:
            raise ValueError(
                "Time range must look like 'start,end', for example '10,60'."
            )
        start, end = float(parts[0]), float(parts[1])
        if not (0.0 <= start < end):
            raise ValueError("Time range must satisfy 0 <= start < end.")
        return (start, end)

    # ----- run
    def _run(self):
        eeg_file = self.eeg_edit.text().strip()
        dbs_file = self.dbs_edit.text().strip()
        sub_id = self.sub_edit.text().strip()
        block = self.block_edit.text().strip()
        out_dir = self.out_edit.text().strip() or "outputs"

        # Friendly validation BEFORE touching the heavy pipeline.
        if not eeg_file or not Path(eeg_file).exists():
            self._error("Please choose a valid EEG file.")
            return
        if not dbs_file or not Path(dbs_file).exists():
            self._error("Please choose a valid DBS JSON file.")
            return
        if not sub_id or not block:
            self._error("Please fill in both a Subject ID and a Block/session label.")
            return
        try:
            time_range = self._parse_time_range()
        except ValueError as exc:
            self._error(str(exc))
            return

        use_gui = self.manual_check.isChecked()
        save_plots = self.plots_check.isChecked()

        self.status.setText("Running… this can take a moment.")
        self.QtWidgets.QApplication.processEvents()

        # Defer the heavy imports to here so the window opens instantly.
        try:
            import matplotlib

            # The manual slider window needs a Qt backend; otherwise headless Agg.
            matplotlib.use("QtAgg" if use_gui else "Agg")
            from .core import sync_run, save_run_metadata
        except Exception as exc:  # pragma: no cover
            self._error(f"Could not load the synchronization engine:\n{exc}")
            self.status.setText("Failed to start. See the message box.")
            return

        try:
            res = sync_run(
                sub_id=sub_id,
                block=block,
                eeg_file=Path(eeg_file),
                dbs_file=Path(dbs_file),
                time_range=time_range,
                use_gui=use_gui,
            )
            out_path = Path(out_dir)
            meta_path = save_run_metadata(res, out_path)

            if save_plots:
                self._save_plots(res, out_path)

            self.status.setText(f"Done. Metadata saved to:\n{meta_path}")
            self._info(
                "Synchronization complete.\n\n"
                f"Subject: {res.sub_id}  Block: {res.block}\n"
                f"EEG sync: {res.eeg_sync_s:.3f} s\n"
                f"DBS sync: {res.dbs_sync_s:.3f} s\n\n"
                f"Results saved in:\n{out_path.resolve()}"
            )
        except Exception as exc:
            logger.exception("GUI run failed")
            self.status.setText("Synchronization failed. See the message box.")
            self._error(
                "Synchronization could not be completed:\n\n"
                f"{exc}\n\n"
                "Tip: check that the EEG and DBS files match and that the "
                "time range (if any) is inside the recording."
            )

    def _save_plots(self, res, out_path: Path) -> None:
        import numpy as np

        from . import plotting

        plots_dir = out_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        eeg_t = res.synchronized_eeg.times
        eeg_y = res.synchronized_eeg.get_data()[0]
        dbs_y = res.synchronized_dbs["TimeDomainData"].values
        dbs_fs = res.metadata["fs"]["dbs"]
        dbs_t = np.arange(len(dbs_y)) / dbs_fs

        plotting.plot_eeg_dbs_overlay(
            eeg_t, eeg_y, dbs_t, dbs_y,
            outdir=plots_dir, sub_id=res.sub_id, block=res.block, show=False,
        )
        plotting.plot_dbs_artifact(
            dbs_y, dbs_fs, res.dbs_sync_idx,
            outdir=plots_dir, sub_id=res.sub_id, block=res.block, show=False,
        )

    def show(self):
        self.window.show()


def main(argv=None) -> int:
    """Open the launcher window. Returns a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        QtWidgets, QtCore = _import_qt()
    except RuntimeError as exc:
        # No Qt available: print a clear, non-technical message.
        sys.stderr.write("\n" + str(exc) + "\n")
        return 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        ["dbs-eeg-sync-gui"]
    )
    launcher = _LauncherWindow(QtWidgets, QtCore)
    launcher.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
