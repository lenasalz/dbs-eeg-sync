EEG–DBS Synchronization Toolbox

This repository provides open, reproducible code for synchronizing EEG recordings with deep brain stimulation (DBS) signals based on stimulation artifacts. The package enables precise temporal alignment between cortical EEG and subcortical local field potentials (LFPs) from DBS systems and was developed in the context of our Brain Stimulation manuscript.

⸻

Overview

The repository is organized as a modular Python package, dbs_eeg_sync, with clear separation between computation, visualization, and user interaction layers.

.
├── dbs_eeg_sync/
│   ├── core.py                    # orchestration logic (sync_run)
│   ├── synchronizer.py            # signal alignment and resampling
│   ├── sync_artifact_finder.py    # artifact detection routines
│   ├── data_loader.py             # EEG/DBS data import utilities
│   ├── power_calculator.py        # band-power computation
│   ├── plotting.py                # plotting utilities (headless support)
│   ├── cli.py                     # command-line interface
│   ├── gui.py                     # optional manual sync GUI
│   └── __init__.py                # public API exports
├── tests/                         # unit tests
├── config/                        # JSON/YAML configuration files
├── data/                          # example EEG/DBS input data
├── notebooks/                     # Jupyter notebooks (examples)
└── outputs/                       # generated plots and metadata


⸻

Installation

### 1. Clone the repository

```bash
git clone https://github.com/lenasalz/dbs-eeg-sync
cd dbs-eeg-sync
```

### 2. Install dependencies

**Option A: Using `uv` (Recommended)**

[`uv`](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or: pip install uv

# Install the package
uv sync
```

**Option B: Using `pip` (Traditional)**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

**Option C: Using `conda`**

```bash
conda create -n dbs-eeg-sync python=3.12
conda activate dbs-eeg-sync
pip install -e .
```


⸻
Running the Synchronization

### Quick Test with Example Data

```bash
# If using uv:
uv run dbs-eeg-sync --test --plots --headless

# If using pip/conda (with activated environment):
dbs-eeg-sync --test --plots --headless
```

This produces synchronized outputs, metadata JSONs, and headless plots under `outputs/`.

### Using Your Own Data

```bash
dbs-eeg-sync \
  --sub-id P01 \
  --block baseline \
  --eeg-file /path/to/eeg.set \
  --dbs-file /path/to/dbs.json \
  --time-range 0,120 \
  --plots --headless --output-dir outputs
```

**Note:** If you installed with `uv sync`, prefix commands with `uv run` or activate the environment first:
```bash
source .venv/bin/activate  # Then run dbs-eeg-sync directly
```


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

Batch Processing

Use a CSV manifest to synchronize multiple subjects automatically:

sub_id,block,eeg_file,dbs_file,start_sec,end_sec
S01,B1,data/eeg_example.set,data/dbs_example.json,0,60
S02,Baseline,/abs/path/eeg_S02.set,/abs/path/dbs_S02.json,10,120

Run:

dbs-eeg-sync --manifest config/manifest.csv --plots --headless


⸻

Output
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

Module Overview

| Module | Description |
|--------|-------------|
| `core.py` | Orchestrates full synchronization (non-interactive) via `sync_run`. |
| `synchronizer.py` | Core signal alignment and resampling logic (no I/O). |
| `sync_artifact_finder.py` | Artifact detection in EEG and DBS data. |
| `data_loader.py` | EEG and DBS data import helpers (EEGLAB .set, JSON, and more). |
| `power_calculator.py` | Sample-wise band-power calculation for artifact detection. |
| `plotting.py` | Headless plotting utilities for DBS artifacts, EEG power, and overlays. |
| `cli.py` | Command-line interface for batch and config-driven execution. |
| `gui.py` | Optional manual synchronization GUI (requires PyQt). |

Example Jupyter notebooks are available in the `notebooks/` directory but not included in the installable package.
⸻

Citation

If you use this software in your research, please cite:

```bibtex
@article{salzmann2025eegdbs,
  title={Synchronizing EEG with Intracranial DBS Electrode Recordings for Neurophysiological Research},
  author={Salzmann, Lena and others},
  journal={},
  year={2025},
  note={Manuscript in review}
}
```

For more citation formats, see [`CITATION.cff`](CITATION.cff).

⸻

Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on:
- Reporting bugs and requesting features
- Setting up a development environment
- Code style and testing requirements
- Submitting pull requests

⸻

License and Acknowledgements

This code is distributed under the [BSD 3-Clause License](LICENSE). 

Developed at **ETH Zurich**, Department of Health Sciences and Technology, Rehabilitation Engineering Laboratory.

**Contact:** [lena.salzmann@hest.ethz.ch](mailto:lena.salzmann@hest.ethz.ch)