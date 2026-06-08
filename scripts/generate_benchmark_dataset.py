#!/usr/bin/env python3
"""
generate_benchmark_dataset.py
=============================
Generate a small, fully synthetic **benchmark dataset** with ground-truth
synchronization-artifact times, for validating EEG-DBS synchronization /
artifact-detection algorithms. No patient-derived data is used.

The benchmark spans the scenarios characterized in the paper:
  * cohorts/sampling: PD (2000 Hz EEG, 130 Hz stim) and epilepsy
    (256 Hz EEG, 145 Hz stim -> stimulation fundamental above the EEG Nyquist);
  * artifact polarity in the EEG band power: drop vs. spike;
  * data quality: clean vs. noisy/movement;
  * a low-amplitude (harder) case and a double-artifact (drift) case.

Each recording is written as an EEGLAB ``.set`` (EEG) + a Percept-style
``.json`` (LFP), and every recording's ground-truth artifact time(s) are
recorded in ``benchmark/manifest.csv``.

Run from the repository root (requires ``eeglabio`` to write .set):

    .venv/bin/python scripts/generate_benchmark_dataset.py

Output:
    benchmark/recordings/<id>/eeg.set
    benchmark/recordings/<id>/dbs.json
    benchmark/manifest.csv

A fixed per-recording RNG seed makes the dataset reproducible.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import mne

mne.set_log_level("ERROR")

LFP_FS = 250.0
LFP_SNAPSHOT_FS = 2.0
DURATION_S = 30.0

# Compact but standard 10-20 montage (posterior/temporal/central included).
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
            "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2"]
# Relative carrier-pickup topography (posterior/temporal strongest), in microvolts.
CH_BASE_UV = {
    "Fp1": 60, "Fp2": 60, "F7": 120, "F3": 50, "Fz": 30, "F4": 60, "F8": 140,
    "T7": 320, "C3": 160, "Cz": 70, "C4": 150, "T8": 360, "P7": 460, "P3": 240,
    "Pz": 180, "P4": 250, "P8": 470, "O1": 560, "Oz": 540, "O2": 520,
}

# id, cohort, eeg_fs, stim_hz, polarity, noise, artifact_s, second_artifact_s, note
SCENARIOS = [
    ("pd_clean_drop",   "PD",  2000, 130, "drop",  "clean", 12.0, None, "baseline drop-polarity"),
    ("pd_clean_spike",  "PD",  2000, 130, "spike", "clean", 15.0, None, "baseline spike-polarity"),
    ("pd_noisy_drop",   "PD",  2000, 130, "drop",  "noisy", 18.0, None, "movement noise"),
    ("pd_noisy_spike",  "PD",  2000, 130, "spike", "noisy", 14.0, None, "movement noise"),
    ("pd_lowamp",       "PD",  2000, 130, "drop",  "clean", 16.0, None, "weak 0.1 mA-like artifact"),
    ("pd_double",       "PD",  2000, 130, "drop",  "clean", 10.0, 25.0, "two events (drift test)"),
    ("epi_clean_drop",  "EPI",  256, 145, "drop",  "clean", 13.0, None, "sub-Nyquist stim (145/256)"),
    ("epi_clean_spike", "EPI",  256, 145, "spike", "clean", 17.0, None, "sub-Nyquist stim (145/256)"),
    ("epi_noisy_drop",  "EPI",  256, 145, "drop",  "noisy", 19.0, None, "sub-Nyquist + noise"),
]

ARTIFACT_DUR_S = 0.165


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec /= np.sqrt(f)
    out = np.fft.irfft(spec, n)
    return out / out.std()


def make_eeg(params: dict, rng: np.random.Generator) -> mne.io.RawArray:
    eeg_fs = params["eeg_fs"]
    stim = params["stim_hz"]
    n = int(DURATION_S * eeg_fs)
    t = np.arange(n) / eeg_fs
    noisy = params["noise"] == "noisy"
    weak = params["note"].startswith("weak")
    amp_scale = 0.25 if weak else 1.0

    # carrier frequency as seen in the EEG: aliased when stim exceeds Nyquist
    nyq = eeg_fs / 2.0
    carrier_hz = stim if stim < nyq else abs(stim - eeg_fs)  # alias fold

    events = [params["artifact_s"]]
    if params["second_artifact_s"] is not None:
        events.append(params["second_artifact_s"])

    data = np.zeros((len(CHANNELS), n))
    floor = (35.0 if noisy else 18.0) * 1e-6
    for i, ch in enumerate(CHANNELS):
        target = CH_BASE_UV[ch] * 1e-6
        pink = pink_noise(n, rng) * floor
        carrier_amp = np.sqrt(2.0) * max(target, 5e-6)
        phase = rng.uniform(0, 2 * np.pi)

        env = np.ones(n)
        burst = np.zeros(n)
        for a_t in events:
            m = (t >= a_t) & (t < a_t + ARTIFACT_DUR_S)
            bm = np.where(m)[0]
            # The sharp stimulation-amplitude change injects a broadband transient.
            # This is the feature the high-band detector keys on, and it is what
            # makes detection work even when the stimulation carrier is above the
            # EEG Nyquist (epilepsy: 145 Hz carrier aliases to ~111 Hz, outside
            # the 120-127 Hz band). Present for both polarities.
            burst[bm] += carrier_amp * (0.9 * amp_scale) * rng.standard_normal(bm.size)
            if params["polarity"] == "drop":
                # also suppress the in-band carrier (visible as a power drop where
                # the carrier lies within the detection band, i.e. the PD cohort)
                env[m] = 1.0 - 0.6 * amp_scale

        sig = pink + carrier_amp * env * np.sin(2 * np.pi * carrier_hz * t + phase) + burst
        if noisy:  # movement: slow drifts + occasional broadband bumps (not at the artifact)
            sig += pink_noise(n, rng) * (80e-6) * np.exp(-((t - 7.0) ** 2) / 4.0)
        data[i] = sig

    info = mne.create_info(ch_names=CHANNELS, sfreq=float(eeg_fs), ch_types="eeg")
    return mne.io.RawArray(data, info)


def make_dbs(params: dict, rng: np.random.Generator) -> dict:
    n = int(DURATION_S * LFP_FS)
    t = np.arange(n) / LFP_FS
    weak = params["note"].startswith("weak")
    sig = pink_noise(n, rng) * 8.0

    events = [params["artifact_s"]]
    if params["second_artifact_s"] is not None:
        events.append(params["second_artifact_s"])
    peak = 30.0 if weak else 65.0
    for a_t in events:
        a0 = int(a_t * LFP_FS)
        w = max(int(0.04 * LFP_FS), 1)
        k = np.arange(-3 * w, 3 * w + 1)
        sig[a0 + k[0]: a0 + k[-1] + 1] += peak * np.exp(-(k**2) / (2 * (w / 2) ** 2))
    sig = np.clip(sig, None, peak - 1.0)
    sig[int(events[0] * LFP_FS)] = peak  # guarantee global positive max at first event

    n_pk = n // 125
    td = {
        "Channel": "ONE_THREE_RIGHT", "Gain": 227, "SampleRateInHz": int(LFP_FS),
        "FirstPacketDateTime": "2024-01-01T09:05:00.000Z",
        "GlobalSequences": ",".join(str(s) for s in range(n_pk)),
        "GlobalPacketSizes": ",".join("125" for _ in range(n_pk)),
        "TicksInMses": ",".join(str(196000 + 500 * p) for p in range(n_pk)),
        "TimeDomainData": [round(float(x), 6) for x in sig],
    }
    snaps = []
    n_sn = int(DURATION_S * LFP_SNAPSHOT_FS)
    for j in range(n_sn):
        ts = j / LFP_SNAPSHOT_FS
        in_art = any(a <= ts < a + 1.0 for a in events)
        snaps.append({"Seq": 2 + j, "TicksInMs": 196000 + int(ts * 1000),
                      "Right": {"LFP": int(max(0, rng.normal(260, 25) * (0.6 if in_art else 1.0))),
                                "mA": round(2.8 - (0.5 if in_art else 0.0), 1)},
                      "Left": {"LFP": 0, "mA": 2.6}})
    return {
        "PatientInformation": {"Initial": {"PatientFirstName": "Synthetic",
                                           "PatientId": "BENCH", "Diagnosis":
                                           f"DiagnosisTypeDef.{'ParkinsonsDisease' if params['cohort']=='PD' else 'Epilepsy'}"}},
        "DeviceInformation": {"Initial": {"Neurostimulator": "Percept PC",
                                          "NeurostimulatorSerialNumber": "SYNTHETIC0000"}},
        "BrainSenseTimeDomain": [td],
        "BrainSenseLfp": [{"Channel": "ONE_THREE_RIGHT", "SampleRateInHz": int(LFP_SNAPSHOT_FS),
                           "TherapySnapshot": {"Right": {"RateInHertz": params["stim_hz"]}},
                           "LfpData": snaps}],
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    bench = root / "benchmark"
    rec_dir = bench / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, sc in enumerate(SCENARIOS):
        (rid, cohort, eeg_fs, stim, pol, noise, a1, a2, note) = sc
        params = {"cohort": cohort, "eeg_fs": eeg_fs, "stim_hz": stim, "polarity": pol,
                  "noise": noise, "artifact_s": a1, "second_artifact_s": a2, "note": note}
        rng = np.random.default_rng(1000 + idx)
        d = rec_dir / rid
        d.mkdir(exist_ok=True)
        make_eeg(params, rng).export(str(d / "eeg.set"), fmt="eeglab", overwrite=True)
        with (d / "dbs.json").open("w", encoding="utf-8") as f:
            json.dump(make_dbs(params, rng), f, ensure_ascii=False)
        rows.append({
            "recording_id": rid, "cohort": cohort, "eeg_file": f"recordings/{rid}/eeg.set",
            "dbs_file": f"recordings/{rid}/dbs.json", "eeg_fs_hz": eeg_fs, "lfp_fs_hz": int(LFP_FS),
            "stim_freq_hz": stim, "eeg_band_polarity": pol, "data_quality": noise,
            "artifact_time_s": a1, "second_artifact_time_s": "" if a2 is None else a2,
            "notes": note,
        })
        print(f"  [{idx+1}/{len(SCENARIOS)}] {rid}: artifact @ {a1}s"
              + (f" & {a2}s" if a2 else ""))

    manifest = bench / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} recordings + manifest -> {manifest}")
    print("All synthetic; no patient-derived data.")


if __name__ == "__main__":
    main()
