#!/usr/bin/env python3
"""
evaluate_benchmark.py
=====================
Run the EEG-DBS synchronization detector over the synthetic benchmark dataset
and report detection error against the ground-truth artifact times in
``benchmark/manifest.csv``.

This serves two purposes:
  * a sanity check that the bundled detector finds the labeled artifacts, and
  * a reusable harness: point your own detector at the manifest and compare.

Run from the repository root (after generate_benchmark_dataset.py):

    .venv/bin/python scripts/evaluate_benchmark.py

What "within tolerance" means here:
  * LFP: 1 sample at 250 Hz (4 ms) -- the detector picks the transient peak, so
    LFP localization is effectively exact. This carries the precision.
  * EEG: 1.0 s -- a *correct-event* criterion, not a precision metric. The EEG
    detector marks the band-power onset/midpoint after a fixed 301-sample
    Savitzky-Golay smoothing; at 256 Hz (epilepsy) that window spans ~1.2 s, so
    the marked time can sit several hundred ms from the labeled event start even
    when the right event is found. The raw error is always printed.
  * Recordings with two events (drift case): a detection matching *either*
    labeled event counts.
"""
from __future__ import annotations

import csv
from pathlib import Path

from dbs_eeg_sync.core import sync_run

LFP_TOL_S = 0.004
EEG_TOL_S = 1.0


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    bench = root / "benchmark"
    manifest = bench / "manifest.csv"
    if not manifest.exists():
        raise SystemExit("benchmark/manifest.csv not found - run generate_benchmark_dataset.py first.")

    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"{'recording':<18}{'truth':>7}{'EEG@':>9}{'EEGerr':>9}{'LFP@':>8}{'LFPerr':>9}  result")
    print("-" * 74)
    n_pass = 0
    summary: dict[str, list[bool]] = {}
    for r in rows:
        truths = [float(r["artifact_time_s"])]
        if r["second_artifact_time_s"]:
            truths.append(float(r["second_artifact_time_s"]))
        truth = truths[0]
        try:
            res = sync_run(
                sub_id=r["recording_id"], block="bench",
                eeg_file=bench / r["eeg_file"], dbs_file=bench / r["dbs_file"],
                time_range=None, block_index=0, headless=True, auto_crop=False,
            )
            # match against the nearest labeled event (drift case has two)
            eeg_err = min(abs(res.eeg_sync_s - g) for g in truths)
            lfp_err = min(abs(res.dbs_sync_s - g) for g in truths)
            ok = (eeg_err <= EEG_TOL_S) and (lfp_err <= LFP_TOL_S)
            n_pass += ok
            print(f"{r['recording_id']:<18}{truth:>6.1f}s{res.eeg_sync_s:>8.2f}s"
                  f"{eeg_err*1000:>7.0f}ms{res.dbs_sync_s:>7.2f}s{lfp_err*1000:>7.0f}ms  "
                  f"{'PASS' if ok else 'CHECK'}")
        except Exception as exc:  # detection genuinely failed for this recording
            ok = False
            print(f"{r['recording_id']:<18}{truth:>6.1f}s{'--':>9}{'--':>9}{'--':>8}{'--':>9}  ERROR: {exc}")
        summary.setdefault(r["data_quality"], []).append(ok)

    print("-" * 74)
    print(f"Overall: {n_pass}/{len(rows)} within tolerance "
          f"(EEG <= {EEG_TOL_S*1000:.0f} ms, LFP <= {LFP_TOL_S*1000:.0f} ms)")
    for q, oks in sorted(summary.items()):
        print(f"  {q:<8}: {sum(oks)}/{len(oks)}")
    print("\nNote: 'noisy' recordings may need manual correction (as in the paper); "
          "a CHECK there is expected, not a bug.")


if __name__ == "__main__":
    main()
