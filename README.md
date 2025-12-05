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

### Using a Config File

Create your own config file (JSON or YAML) with your parameters:

```bash
dbs-eeg-sync --config path/to/your_config.json
```

**Precedence:** defaults < config file < CLI flags

This means you can override any config file parameter with CLI flags.

### Configuration Parameters Reference

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sub_id` | string | Yes | Subject identifier (e.g., "P01", "S01") |
| `block` | string | Yes | Recording block/session label (e.g., "baseline", "B1") |
| `eeg_file` | string | Yes | Path to EEG file (supports EEGLAB `.set`, EDF, XDF, etc.) |
| `dbs_file` | string | Yes | Path to DBS JSON file from recording device |
| `block_index` | integer | No | DBS recording index (0-based) if JSON contains multiple recordings (default: `0` - first recording) |
| `time_range` | string or array | No | Time window for artifact detection: `"start,end"` in seconds (e.g., `"10,60"` or `[10, 60]`). If omitted, uses full recording. |
| `output_dir` | string | No | Directory for outputs (default: `"outputs"`) |
| `plots` | boolean | No | Generate visualization plots (default: `false`) |
| `headless` | boolean | No | Use non-interactive backend for plots, no GUI windows (default: `false`) |
| `gui` | boolean | No | Enable manual GUI selection for sync points (default: `false`). Cannot be combined with `headless`. |
| `verbose` | boolean | No | Enable detailed logging output (default: `false`) |

### Example: JSON Config

`config/default.json`:

```json
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
```

Run with:

```bash
dbs-eeg-sync --config config/default.json
```

### Example: YAML Config

`config/default.yml` (requires `pyyaml`):

```yaml
sub_id: S01
block: B1
eeg_file: data/eeg_example.set
dbs_file: data/dbs_example.json
time_range: [10, 60]
output_dir: outputs
plots: true
headless: true
```

Run with:

```bash
dbs-eeg-sync --config config/default.yml
```


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

The tool generates three types of outputs:

### 1. Metadata JSON

A timestamped JSON file containing synchronization results and provenance information:

- **Sync indices**: Sample indices in the **original (uncropped)** data where the artifact was detected
  - `eeg_sync_idx`, `eeg_sync_s`: EEG artifact location (sample index and seconds)
  - `dbs_sync_idx`, `dbs_sync_s`: DBS artifact location (sample index and seconds)
- **Data properties**: Sample rates, file paths, channel names
- **Detection parameters**: Artifact frequency band, time range used
- **Provenance**: Package versions, git commit (if available), timestamps

**Note:** The sync indices always refer to the original data, even if you specified a `time_range` for artifact detection. This ensures reproducibility.

### 2. Plots (optional, with `--plots`)

Visualization plots saved under `outputs/plots/`:
- EEG-DBS overlay after synchronization
- DBS artifact detection plot
- EEG power spectrum plots (if applicable)

### 3. Logs

Detailed execution logs written to `outputs/run.log`

### Example Output Structure

```
outputs/
├── 20251028_114817_metadata_S01_B1.json
├── plots/
│   ├── 20251028_114817_syncDBS_S01_B1.png
│   ├── 20251028_114817_eeg_dbs_overlay_S01_B1.png
└── run.log
```

### Example Metadata JSON

A complete example metadata file is available at [`outputs/example_metadata.json`](outputs/example_metadata.json).

Key fields (simplified):

```json
{
  "sub_id": "S01",
  "block": "B1",
  "eeg_sync_idx": 12500,
  "eeg_sync_s": 10.0,
  "dbs_sync_idx": 3125,
  "dbs_sync_s": 10.0,
  "eeg_fs": 1250.0,
  "dbs_fs": 312.5,
  "channel": "T8",
  "artifact_kind": "drop",
  "time_range": [10, 60],
  "created_utc": "2025-11-03T10:30:00Z",
  "versions": {
    "dbs_eeg_sync": "0.1.0",
    "numpy": "1.26.0",
    "mne": "1.5.0"
  }
}
```


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

⸻

Jupyter Notebooks

The `notebooks/` directory contains interactive Jupyter notebooks demonstrating the synchronization methods, validation analyses, and publication figure generation. These notebooks are useful for:

- **Understanding the method**: Step-by-step demonstrations of artifact detection and synchronization
- **Validation**: Reproducing accuracy metrics and validation analyses
- **Publication figures**: Generating all manuscript figures

### Key Notebooks

- **`publication_figures.ipynb`**: Generates all manuscript figures in publication quality
- **`sync_artifact_detection.ipynb`**: Core method demonstration
- **`validation_metrics.ipynb`**: Method validation and accuracy metrics
- **`sync_drop_detection.ipynb`**: Drop-type artifact handling
- **`test_sync_methods.ipynb`**: Method comparison and parameter analysis

See [`notebooks/README.md`](notebooks/README.md) for detailed descriptions of all notebooks.

**Note:** Notebooks are not included in the installable package but are available in the GitHub repository.

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