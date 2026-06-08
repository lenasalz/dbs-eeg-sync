#!/usr/bin/env python3
"""
generate_synthetic_example_data.py
==================================
Generate fully synthetic, non-identifiable example data for the dbs-eeg-sync
toolbox, replacing the patient-derived files in ``data/``.

The synthetic data is *not* recorded from any person. It is procedurally
generated to mimic the *format*, *patterns*, and *magnitude* of a real
recording closely enough that the synchronization pipeline detects the
artifact exactly as it would on real data:

  * ``data/eeg_example.set``  -- EEGLAB EEG, 2000 Hz, 32-channel 10-20 montage,
    70 s. Each channel is pink (1/f) background EEG plus a stimulation carrier
    at the DBS frequency. At the synchronization moment the carrier amplitude
    is briefly reduced (mimicking the 0.5 mA reduction), producing the
    band-power transient the EEG detector keys on. Carrier magnitude follows a
    realistic topography (posterior/temporal channels strongest).

  * ``data/dbs_example.json`` -- a structurally faithful Medtronic Percept(TM)
    BrainSense(TM) export with **fully fictitious** patient/device metadata.
    ``BrainSenseTimeDomain`` holds 250 Hz LFP (pink background + one clear
    positive transient at the artifact, which the LFP detector picks as the
    global positive peak). ``BrainSenseLfp`` holds the 2 Hz power/amplitude
    snapshots with the stimulation amplitude stepping down by 0.5 mA at the
    artifact.

Run from the repository root:

    .venv/bin/python scripts/generate_synthetic_example_data.py

A fixed RNG seed makes the output reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import mne

mne.set_log_level("ERROR")

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
SEED = 20260608
DURATION_S = 45.0
ARTIFACT_T = 20.0            # synchronization artifact time (within the config 5-40 s window)
ARTIFACT_DUR_S = 0.165       # ~165 ms, matching the measured mean artifact duration
STIM_FREQ_HZ = 125.0         # DBS stimulation frequency (inside the 120-130 Hz band)
CARRIER_REDUCTION = 0.6      # fractional carrier-amplitude drop during the artifact

EEG_FS = 2000.0
DBS_TD_FS = 250.0            # BrainSenseTimeDomain sample rate
DBS_LFP_FS = 2.0            # BrainSenseLfp (power/amplitude) sample rate

# 32-channel 10-20 subset (incl. central/temporal/occipital used for detection)
# value = target channel std in microvolts, reproducing the real topography
# (posterior/temporal large, frontal-midline small).
CH_STD_UV = {
    "Fp1": 158, "Fp2": 165, "AF3": 86, "AF4": 95,
    "F7": 275, "F3": 84, "Fz": 34, "F4": 153, "F8": 351,
    "FC5": 319, "FC1": 108, "FCz": 90, "FC2": 60, "FC6": 355,
    "T7": 524, "C3": 305, "Cz": 127, "C4": 254, "T8": 606,
    "CP5": 413, "CP1": 174, "CP2": 232, "CP6": 460,
    "P7": 716, "P3": 437, "Pz": 325, "P4": 434, "P8": 721,
    "POz": 541, "O1": 935, "Oz": 900, "O2": 864,
}
PINK_FLOOR_UV = 20.0   # physiological background std present on every channel


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-std pink (1/f) noise via spectral shaping."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]                      # avoid div-by-zero at DC
    spectrum = spectrum / np.sqrt(freqs)     # 1/f amplitude -> 1/f^2 power
    out = np.fft.irfft(spectrum, n)
    return out / out.std()


def artifact_envelope(t: np.ndarray) -> np.ndarray:
    """Carrier-amplitude multiplier: 1.0 everywhere, dropping during the artifact."""
    env = np.ones_like(t)
    mask = (t >= ARTIFACT_T) & (t < ARTIFACT_T + ARTIFACT_DUR_S)
    env[mask] = 1.0 - CARRIER_REDUCTION
    return env


# --------------------------------------------------------------------------- #
# EEG (.set)
# --------------------------------------------------------------------------- #
def make_eeg(out_path: Path, rng: np.random.Generator) -> None:
    n = int(DURATION_S * EEG_FS)
    t = np.arange(n) / EEG_FS
    env = artifact_envelope(t)
    ch_names = list(CH_STD_UV)
    data = np.zeros((len(ch_names), n), dtype=np.float64)

    for i, ch in enumerate(ch_names):
        target = CH_STD_UV[ch] * 1e-6                       # to volts
        pink = pink_noise(n, rng) * (PINK_FLOOR_UV * 1e-6)  # background EEG
        # carrier amplitude chosen so total std ~= target (pink + carrier in quadrature)
        carrier_std = np.sqrt(max(target**2 - (PINK_FLOOR_UV * 1e-6) ** 2, (5e-6) ** 2))
        carrier_amp = carrier_std * np.sqrt(2.0)
        phase = rng.uniform(0, 2 * np.pi)
        carrier = carrier_amp * env * np.sin(2 * np.pi * STIM_FREQ_HZ * t + phase)
        data[i] = pink + carrier

    info = mne.create_info(ch_names=ch_names, sfreq=EEG_FS, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    try:
        raw.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass
    raw.export(str(out_path), fmt="eeglab", overwrite=True)
    print(f"  EEG: {len(ch_names)} ch x {n} samp @ {EEG_FS:.0f} Hz -> {out_path}")


# --------------------------------------------------------------------------- #
# DBS (.json)
# --------------------------------------------------------------------------- #
def make_time_domain(rng: np.random.Generator) -> tuple[list, int]:
    """250 Hz LFP: pink background + one positive transient at the artifact."""
    n = int(DURATION_S * DBS_TD_FS)
    t = np.arange(n) / DBS_TD_FS
    sig = pink_noise(n, rng) * 8.0                       # std ~8, like the real LFP
    # sharp positive transient (the synchronization artifact) — must be the
    # global positive maximum so detect_dbs_sync_artifact() selects it.
    a0 = int(ARTIFACT_T * DBS_TD_FS)
    width = max(int(0.04 * DBS_TD_FS), 1)
    k = np.arange(-3 * width, 3 * width + 1)
    transient = 65.0 * np.exp(-(k**2) / (2.0 * (width / 2.0) ** 2))
    lo, hi = a0 + k[0], a0 + k[-1] + 1
    sig[lo:hi] += transient
    sig = np.clip(sig, None, 64.0)                       # keep noise below the peak
    sig[a0] = 65.0                                       # guarantee global positive max
    return [round(float(x), 6) for x in sig], n


def make_lfp_snapshots(n_td: int, rng: np.random.Generator) -> list:
    """2 Hz power/amplitude snapshots with a 0.5 mA step-down at the artifact."""
    n = int(DURATION_S * DBS_LFP_FS)
    base_mA = 2.8
    samples = []
    for j in range(n):
        t_s = j / DBS_LFP_FS
        in_artifact = ARTIFACT_T <= t_s < ARTIFACT_T + 1.0
        mA = round(base_mA - (0.5 if in_artifact else 0.0), 1)
        lfp_power = int(max(0, rng.normal(260, 25) * (0.6 if in_artifact else 1.0)))
        samples.append({
            "Seq": 2 + j,
            "TicksInMs": 196000 + int(t_s * 1000),
            "StatusBytes": "00 00 00 00",
            "Right": {"LFP": lfp_power, "mA": mA},
            "Left": {"LFP": 0, "mA": 2.6},
        })
    return samples


def make_dbs_json(out_path: Path, rng: np.random.Generator) -> None:
    td_data, n_td = make_time_domain(rng)
    n_packets = n_td // 125
    ticks = [196000 + 500 * p for p in range(n_packets)]
    seqs = list(range(n_packets))

    # NOTE: all patient/device identifiers below are fictitious.
    record = {
        "AbnormalEnd": False,
        "FullyReadForSession": True,
        "FeatureInformationCode": "SYNTHETIC",
        "SessionDate": "2024-01-01T09:00:00Z",
        "SessionEndDate": "2024-01-01T09:30:00Z",
        "ProgrammerVersion": "0.0.0-synthetic",
        "PatientInformation": {
            "Initial": {
                "PatientFirstName": "Synthetic", "PatientLastName": "Example",
                "PatientGender": "GenderDef.Unspecified",
                "PatientDateOfBirth": "1970-01-01T00:00:00Z",
                "PatientId": "SYNTH-001", "ClinicianNotes": "",
                "Diagnosis": "DiagnosisTypeDef.ParkinsonsDisease",
            },
            "Final": {
                "PatientFirstName": "Synthetic", "PatientLastName": "Example",
                "PatientGender": "GenderDef.Unspecified",
                "PatientDateOfBirth": "1970-01-01T00:00:00Z",
                "PatientId": "SYNTH-001", "ClinicianNotes": "",
                "Diagnosis": "DiagnosisTypeDef.ParkinsonsDisease",
            },
        },
        "DeviceInformation": {
            "Initial": {
                "Neurostimulator": "Percept PC", "NeurostimulatorModel": "B35200",
                "NeurostimulatorSerialNumber": "SYNTHETIC0000",
                "NeurostimulatorLocation": "InsLocation.ABDOMEN_RIGHT",
                "ImplantDate": "2020-01-01T00:00:00Z",
                "DeviceDateTime": "2024-01-01T09:00:00Z",
            },
            "Final": {
                "Neurostimulator": "Percept PC", "NeurostimulatorModel": "B35200",
                "NeurostimulatorSerialNumber": "SYNTHETIC0000",
                "NeurostimulatorLocation": "InsLocation.ABDOMEN_RIGHT",
                "ImplantDate": "2020-01-01T00:00:00Z",
                "DeviceDateTime": "2024-01-01T09:30:00Z",
            },
        },
        "LeadConfiguration": {
            "Initial": [{
                "Hemisphere": "HemisphereLocationDef.Right",
                "Model": "LeadModelDef.LEAD_B33005",
                "LeadLocation": "LeadLocationDef.Stn",
                "ElectrodeNumber": "InsPort.ZERO_THREE",
            }],
        },
        "BrainSenseTimeDomain": [{
            "Pass": "",
            "GlobalSequences": ",".join(str(s) for s in seqs),
            "GlobalPacketSizes": ",".join("125" for _ in seqs),
            "TicksInMses": ",".join(str(x) for x in ticks),
            "Channel": "ONE_THREE_RIGHT",
            "Gain": 227,
            "FirstPacketDateTime": "2024-01-01T09:05:00.000Z",
            "SampleRateInHz": int(DBS_TD_FS),
            "TimeDomainData": td_data,
        }],
        "BrainSenseLfp": [{
            "Channel": "ONE_THREE_RIGHT",
            "FirstPacketDateTime": "2024-01-01T09:05:00.000Z",
            "SampleRateInHz": int(DBS_LFP_FS),
            "TherapySnapshot": {
                "ActiveGroup": "GroupIdDef.GROUP_A",
                "HighPassFilterInHertz": 1,
                "Right": {
                    "SensingChannel": "SensingChannelDef.ONE_THREE_RIGHT",
                    "RateInHertz": STIM_FREQ_HZ,
                },
            },
            "LfpData": make_lfp_snapshots(n_td, rng),
        }],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  DBS: TimeDomain {n_td} samp @ {DBS_TD_FS:.0f} Hz, "
          f"artifact @ {ARTIFACT_T:.0f}s -> {out_path} ({size_mb:.2f} MB)")


# --------------------------------------------------------------------------- #
def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("Generating synthetic example data (seed "
          f"{SEED}, {DURATION_S:.0f}s, artifact @ {ARTIFACT_T:.0f}s):")
    make_eeg(data_dir / "eeg_example.set", rng)
    make_dbs_json(data_dir / "dbs_example.json", rng)
    print("Done. These files contain no patient-derived data.")


if __name__ == "__main__":
    main()
