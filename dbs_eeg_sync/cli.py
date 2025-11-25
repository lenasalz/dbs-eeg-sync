"""
cli.py — Command-Line Interface for EEG–DBS-LFP Synchronization
------------------------------------------------------------
Implements a fully non-interactive CLI (`dbs-eeg-sync`) for running
synchronization jobs. Supports single-run and batch modes, configuration via
JSON/YAML files, headless plotting, and manifest-driven batch execution.
Precedence: defaults < config < CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .core import sync_run, save_run_metadata

logger = logging.getLogger(__name__)

# ---------- helpers ----------

def _load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"--config not found: {path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with: uv pip install pyyaml"
            ) from e
        return yaml.safe_load(path.read_text())
    # default to JSON
    return json.loads(path.read_text())

def _merge_config(defaults: Dict[str, Any], config: Dict[str, Any], flags: Dict[str, Any]) -> Dict[str, Any]:
    # precedence: defaults < config < flags (when not None)
    merged = {**defaults, **(config or {})}
    for k, v in flags.items():
        if v is not None:
            merged[k] = v
    return merged

def _configure_logging(verbose: bool, outdir: Optional[Path]) -> None:
    root_level = logging.INFO
    pkg_level = logging.DEBUG if verbose else logging.INFO

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(outdir) / "run.log", mode="a"))

    # Force reconfigure in case something configured logging earlier (e.g., imported libs)
    logging.basicConfig(level=root_level, format=fmt, handlers=handlers, force=True)

    # Our package should respect --verbose
    logging.getLogger("dbs_eeg_sync").setLevel(pkg_level)

    # Turn down very chatty libraries that commonly emit DEBUG/INFO noise
    for noisy_name in (
        "matplotlib",
        "matplotlib.font_manager",
        "numba",
        "urllib3",
        "PIL",
        "mne",
    ):
        logging.getLogger(noisy_name).setLevel(logging.WARNING)

def _ensure_headless(headless: bool) -> None:
    if headless or os.environ.get("DISPLAY") is None:
        import matplotlib  # defer import
        try:
            matplotlib.use("Agg")
        except Exception:
            pass

def _parse_time_range(s: Optional[str]) -> Optional[Tuple[float, float]]:
    if not s:
        return None
    a, b = (float(x) for x in s.split(","))
    if not (0.0 <= a < b):
        raise argparse.ArgumentTypeError("--time-range must be 'start,end' with 0 <= start < end")
    return (a, b)

def _jsonify_args(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out

def _result_to_meta(res) -> Dict[str, Any]:
    """
    Produce a JSON-serializable metadata dict from SyncResult.
    Excludes heavy/non-serializable objects (MNE Raw, DataFrames, etc.).
    """
    def _scalar(x):
        # Convert numpy scalars to Python primitives
        try:
            import numpy as _np
            if isinstance(x, (_np.generic,)):
                return x.item()
        except Exception:
            pass
        return x

    meta: Dict[str, Any] = {
        "sub_id": res.sub_id,
        "block": res.block,
        "eeg_file": str(res.eeg_file),
        "dbs_file": str(res.dbs_file),
        "eeg_sync_idx": int(res.eeg_sync_idx),
        "eeg_sync_s": float(res.eeg_sync_s),
        "dbs_sync_idx": int(res.dbs_sync_idx),
        "dbs_sync_s": float(res.dbs_sync_s),
        "eeg_fs": float(res.eeg_fs),
        "dbs_fs": float(res.dbs_fs),
        "eeg_channel": res.channel,
        "artifact_kind": res.artifact_kind,
        "artifact_magnitude": _scalar(res.artifact_magnitude) if res.artifact_magnitude is not None else None,
        "time_range": list(res.time_range) if res.time_range is not None else None,
    }
    # Include select provenance from res.metadata if present (and JSON-friendly)
    if isinstance(res.metadata, dict):
        keep_keys = ("artifact_band_hz", "fs", "block_num")
        for k in keep_keys:
            if k in res.metadata:
                meta[k] = res.metadata[k]
    return meta

def _normalize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    # Parse time_range if the config provided a string or a list
    tr = args.get("time_range")
    if isinstance(tr, str):
        args["time_range"] = _parse_time_range(tr)
    elif isinstance(tr, (list, tuple)):
        if len(tr) == 2:
            args["time_range"] = (float(tr[0]), float(tr[1]))
        elif tr is None:
            args["time_range"] = None
        else:
            raise argparse.ArgumentTypeError(
                "--time-range in config must be 'start,end' or [start, end]"
            )
    # Expand paths from config/flags
    for k in ("eeg_file", "dbs_file", "output_dir"):
        if args.get(k) is not None and not isinstance(args[k], Path):
            args[k] = Path(args[k]).expanduser()
    return args

# ---------- single-job runner ----------

def _run_one(args: Dict[str, Any]) -> int:
    """
    Execute one sync job and optionally save plots/metadata.
    Returns 0 on success, non-zero on failure.
    """
    sub_id = args["sub_id"]
    block = args["block"]
    eeg_file = Path(args["eeg_file"])
    dbs_file = Path(args["dbs_file"])
    time_range = args.get("time_range")
    output_dir = Path(args["output_dir"]) if args.get("output_dir") else Path("outputs")
    plots = bool(args.get("plots"))
    headless = bool(args.get("headless"))
    use_gui = bool(args.get("gui"))

    # Disallow conflicting flags before touching backends
    if use_gui and headless:
        logger.error("Cannot combine --gui with --headless. Remove one of the flags.")
        sys.exit(1)

    # Backend selection: if GUI requested, force an interactive backend and skip headless.
    if use_gui:
        os.environ.setdefault("MPLBACKEND", "TkAgg")
    else:
        # Configure non-interactive backend if needed (must be before any pyplot import)
        _ensure_headless(headless)

    try:
        res = sync_run(
            sub_id=sub_id,
            block=block,
            eeg_file=eeg_file,
            dbs_file=dbs_file,
            time_range=time_range,
            use_gui=use_gui,
        )
        extra = {
            "cli": {
                "sub_id": sub_id,
                "block": block,
                "eeg_file": str(eeg_file),
                "dbs_file": str(dbs_file),
                "time_range": list(time_range) if time_range else None,
                "plots": plots,
                "headless": headless,
                "gui": use_gui,
            }
        }
        meta_path = save_run_metadata(res, output_dir, extra=extra)
        logger.info("Saved metadata: %s", meta_path)

        # Optional plotting
        if plots:
            from . import plotting  # local import so headless backend is set first
            import numpy as np

            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Overlay plot
            eeg_t = res.synchronized_eeg.times
            eeg_y = res.synchronized_eeg.get_data()[0]
            dbs_y = res.synchronized_dbs["TimeDomainData"].values
            dbs_fs = res.metadata["fs"]["dbs"]
            dbs_t = np.arange(len(dbs_y)) / dbs_fs

            plotting.plot_eeg_dbs_overlay(
                eeg_t, eeg_y, dbs_t, dbs_y,
                outdir=plots_dir, sub_id=res.sub_id, block=res.block, show=not headless
            )

            # DBS artifact plot
            plotting.plot_dbs_artifact(
                dbs_y, dbs_fs, res.dbs_sync_idx,
                outdir=plots_dir, sub_id=res.sub_id, block=res.block, show=not headless
            )

        logger.info("Completed: %s %s", sub_id, block)
        return 0
    except Exception as e:
        logger.exception("Job failed: %s %s (%s)", sub_id, block, e)
        return 1



# ---------- argparse / entry point ----------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Synchronize EEG with DBS LFP recordings via stimulation artifact (non-interactive CLI)."
    )
    p.add_argument("--sub-id", help="Subject ID")
    p.add_argument("--block", help="Block/session label")
    p.add_argument("--eeg-file", type=Path, help="Path to EEG file (e.g., .set)")
    p.add_argument("--dbs-file", type=Path, help="Path to DBS JSON file")
    p.add_argument("--time-range", type=str, help="start,end (seconds); e.g. 120,200")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Root output directory")
    p.add_argument("--plots", action="store_true", help="Save plots (and show unless --headless)")
    p.add_argument("--headless", action="store_true", help="Do not open any GUI windows; use non-interactive matplotlib backend")
    p.add_argument("--force", action="store_true", help="Overwrite outputs if they exist (reserved)")
    p.add_argument("--verbose", action="store_true", help="Verbose logs")
    p.add_argument("--config", type=Path, help="Path to JSON or YAML config file")
    p.add_argument("--manifest", type=Path, help="CSV with columns: sub_id,block,eeg_file,dbs_file,start_sec,end_sec")
    p.add_argument("--test", action="store_true", help="Use bundled example data under ./data")
    p.add_argument("--gui", action="store_true", help="Enable manual GUI selection for synchronization (requires display; default off).")
    p.add_argument("--no-crop", action="store_true", help="Disable automatic trimming to a common window.")
    return p

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    ns = parser.parse_args(argv)

    defaults: Dict[str, Any] = {
        "output_dir": Path("outputs"),
        "plots": False,
        "headless": False,
        "force": False,
        "verbose": False,
        "time_range": None,
    }

    # Load config (optional)
    config: Dict[str, Any] = {}
    if ns.config:
        config = _load_config_file(ns.config)

    # Flags to dict
    flags: Dict[str, Any] = {
        "sub_id": ns.sub_id,
        "block": ns.block,
        "eeg_file": ns.eeg_file,
        "dbs_file": ns.dbs_file,
        "time_range": _parse_time_range(ns.time_range),
        "output_dir": ns.output_dir,
        "plots": ns.plots,
        "headless": ns.headless,
        "force": ns.force,
        "verbose": ns.verbose,
        "gui": ns.gui,
        "test": ns.test,
    }

    args = _normalize_args(_merge_config(defaults, config, flags))

    # Configure logging
    _configure_logging(args["verbose"], Path(args["output_dir"]))

    # Example-data convenience
    if args.get("test"):
        args["eeg_file"] = args.get("eeg_file") or Path("data/eeg_example.set")
        args["dbs_file"] = args.get("dbs_file") or Path("data/dbs_example.json")
        args["sub_id"] = args.get("sub_id") or "example"
        args["block"] = args.get("block") or "example"

    # If manifest is provided, run batch mode
    if ns.manifest:
        manifest = ns.manifest
        if not manifest.exists():
            logger.error("Manifest not found: %s", manifest)
            return 2
        failures = 0
        with manifest.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                job = dict(args)  # copy shared args
                job["sub_id"] = row.get("sub_id") or job.get("sub_id")
                job["block"] = row.get("block") or job.get("block")
                job["eeg_file"] = Path(row["eeg_file"]) if row.get("eeg_file") else job.get("eeg_file")
                job["dbs_file"] = Path(row["dbs_file"]) if row.get("dbs_file") else job.get("dbs_file")
                if row.get("start_sec") and row.get("end_sec"):
                    job["time_range"] = (float(row["start_sec"]), float(row["end_sec"]))
                rc = _run_one(job)
                failures += int(rc != 0)
        return 1 if failures else 0

    # Single job mode
    required = ("sub_id", "block", "eeg_file", "dbs_file")
    missing = [k for k in required if not args.get(k)]
    if missing:
        parser.error(
            f"Missing required arguments: {', '.join(missing)}. "
            f"Provide flags or use --test for bundled example data."
        )
    return _run_one(args)

if __name__ == "__main__":
    raise SystemExit(main())