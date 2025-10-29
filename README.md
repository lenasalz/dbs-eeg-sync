EEG–DBS Synchronization Toolbox

This repository provides open, reproducible code for synchronizing EEG recordings with deep brain stimulation (DBS) signals based on stimulation artifacts. The package enables precise temporal alignment between cortical EEG and subcortical local field potentials (LFPs) from DBS systems and was developed in the context of our Brain Stimulation manuscript.

⸻

🧩 Overview

The repository is organized as a modular Python package, dbs_eeg_sync, with clear separation between computation, visualization, and user interaction layers.

.
├── dbs_eeg_sync
│   ├── core.py              # orchestration logic (sync_run)
│   ├── plotting.py          # plotting utilities (headless support)
│   ├── cli.py               # non-interactive command-line interface
│   └── __init__.py
├── source
│   ├── synchronizer.py      # computational core (alignment, resampling)
│   ├── data_loader.py       # EEG/DBS data import utilities
│   └── sync_artefact_finder.py  # artifact detection routines
├── config                   # JSON/YAML configuration files
├── data                     # example EEG/DBS input data
└── outputs                  # generated plots and metadata


⸻

⚙️ Installation

1. Clone the repository

git clone https://github.com/<your-org>/dbs-eeg-sync.git
cd dbs-eeg-sync

2. Set up your environment

We recommend using uv for lightweight, reproducible environments:

uv sync
uv pip install -e .


⸻

🧠 Running the Synchronization

Option 1 — Using Example Data

A minimal test run with example EEG and DBS recordings:

dbs-eeg-sync --test --plots --headless

This produces synchronized outputs, metadata JSONs, and headless plots under outputs/.

Option 2 — Using Your Own Data

Provide paths directly:

dbs-eeg-sync \
  --sub-id P01 \
  --block baseline \
  --eeg-file /path/to/eeg.set \
  --dbs-file /path/to/dbs.json \
  --time-range 0,120 \
  --plots --headless --output-dir outputs


⸻

⚙️ Configuration via JSON/YAML

You can store parameters in a config file instead of typing them each time.

Example config/default.json:

{
  "sub_id": "S01",
  "block": "B1",
  "eeg_file": "data/eeg_example.set",
  "dbs_file": "data/dbs_example.json",
  "time_range": "10,60",
  "output_dir": "outputs",
  "plots": true,
  "headless": true
}

Run:

dbs-eeg-sync --config config/default.json

Or use YAML (requires pyyaml):

sub_id: S01
block: B1
eeg_file: data/eeg_example.set
dbs_file: data/dbs_example.json
time_range: [10, 60]
plots: true
headless: true


⸻

📦 Batch Processing

Use a CSV manifest to synchronize multiple subjects automatically:

sub_id,block,eeg_file,dbs_file,start_sec,end_sec
S01,B1,data/eeg_example.set,data/dbs_example.json,0,60
S02,Baseline,/abs/path/eeg_S02.set,/abs/path/dbs_S02.json,10,120

Run:

dbs-eeg-sync --manifest config/manifest.csv --plots --headless


⸻

🖼 Output
	•	Plots: Saved under outputs/plots/
	•	Metadata: JSON file containing synchronization parameters and provenance
	•	Logs: Written to outputs/run.log

Example files:

outputs/
├── 20251028_114817_metadata_S01_B1.json
├── plots/
│   ├── 20251028_114817_syncDBS_S01_B1.png
│   ├── 20251028_114817_eeg_dbs_overlay_S01_B1.png
└── run.log


⸻

🧩 Module Overview

Module	Description
core.py	Orchestrates full synchronization (non-interactive) via sync_run.
plotting.py	Headless plotting utilities for DBS artifacts, EEG power, and overlays.
cli.py	Command-line interface for batch and config-driven execution.
synchronizer.py	Core signal alignment and resampling logic (no I/O).
data_loader.py	EEG and DBS data import helpers (EEGLAB .set, JSON).
sync_artefact_finder.py	Artifact detection in EEG and DBS data.

Example Jupyter notebooks are available in the repository but not included in the installable package”
⸻

📘 Citation

If you use this code, please cite our upcoming Brain Stimulation manuscript:

Salzmann L, et al. (2025). Synchronizing EEG with Intracranial DBS Electrode Recordings for Neurophysiological Research. Brain Stimulation.

⸻

🔬 License and Acknowledgements

This code is distributed under an open-source license (to be defined). Developed at the ETH Zurich, Department for Health Science and Technology.

Contact: [Your email here]