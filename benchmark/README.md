# Synthetic EEG–DBS synchronization benchmark

A small, fully **synthetic** benchmark dataset with **ground-truth
synchronization-artifact times**, for validating EEG–DBS synchronization /
artifact-detection algorithms. It contains **no patient-derived data** — every
recording is procedurally generated — so it can be shared and redistributed
freely.

## Generate it

```bash
python scripts/generate_benchmark_dataset.py   # requires `eeglabio` to write .set
```

This writes:

```
benchmark/
├── manifest.csv                 # ground-truth labels (one row per recording)
└── recordings/<id>/
    ├── eeg.set                  # EEGLAB EEG (2000 Hz PD / 256 Hz epilepsy)
    └── dbs.json                 # Percept-style LFP (250 Hz) + 2 Hz snapshots
```

The generator is seeded, so the dataset is reproducible bit-for-bit.

## Evaluate a detector

```bash
python scripts/evaluate_benchmark.py
```

This runs the bundled detector over every recording and reports the error
against ground truth. To benchmark **your own** detector, read `manifest.csv`,
run your method on each `eeg_file` / `dbs_file`, and compare your detected
times to `artifact_time_s` (and `second_artifact_time_s` where present).

## Scenarios

The benchmark spans the axes characterized in the paper:

| recording | cohort | EEG fs | stim | EEG polarity | quality | event(s) |
|-----------|--------|-------:|-----:|--------------|---------|----------|
| `pd_clean_drop`   | PD  | 2000 Hz | 130 Hz | drop  | clean | 12 s |
| `pd_clean_spike`  | PD  | 2000 Hz | 130 Hz | spike | clean | 15 s |
| `pd_noisy_drop`   | PD  | 2000 Hz | 130 Hz | drop  | noisy | 18 s |
| `pd_noisy_spike`  | PD  | 2000 Hz | 130 Hz | spike | noisy | 14 s |
| `pd_lowamp`       | PD  | 2000 Hz | 130 Hz | drop  | clean | 16 s (weak, ~0.1 mA-like) |
| `pd_double`       | PD  | 2000 Hz | 130 Hz | drop  | clean | 10 s + 25 s (drift test) |
| `epi_clean_drop`  | EPI |  256 Hz | 145 Hz | drop  | clean | 13 s |
| `epi_clean_spike` | EPI |  256 Hz | 145 Hz | spike | clean | 17 s |
| `epi_noisy_drop`  | EPI |  256 Hz | 145 Hz | drop  | noisy | 19 s |

Notes on the modeling:
- The artifact is a brief (~165 ms) **broadband transient** from the sharp
  stimulation-amplitude change, plus (for "drop") a suppression of the in-band
  stimulation carrier. The broadband transient is what makes detection work for
  the **epilepsy** cohort, where the 145 Hz stimulation carrier exceeds the
  256 Hz EEG Nyquist (aliasing to ~111 Hz, outside the 120–127 Hz detection
  band).
- "polarity" labels the sign of the band-power change; it is a phase/referencing
  effect in real data, where the same event can read as a spike or a drop.

## Interpreting results

- **LFP localization is effectively exact** (the detector picks the induced
  transient's peak) — this carries the precision.
- **EEG localization** marks the band-power onset/midpoint and is generally
  within ~50–250 ms of the labeled event on clean recordings (at both 2000 Hz
  and 256 Hz, now that the Savitzky–Golay smoothing uses a fixed *duration*
  window rather than a fixed sample count). The evaluation uses a 1 s EEG
  tolerance (a correct-event criterion) and a 4 ms LFP tolerance, and always
  prints the raw error.
- The **noisy** recordings deliberately include movement-like interference;
  occasional misses there are expected and mirror the manual-correction cases
  reported in the paper — not a defect of the benchmark.

## License & citation

Released under the repository's BSD-3-Clause license. If you use this benchmark,
please cite the software (see `CITATION.cff`).
