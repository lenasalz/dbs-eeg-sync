# EEG-DBS Synchronization Pipeline

**EEG-DBS Synchronization Pipeline** is a Python-based tool designed to synchronize EEG (Electroencephalography) and DBS (Deep Brain Stimulation) BrainSense :tm: data. It enables efficient data loading, peak detection, alignment, and visualization of EEG and DBS signals for neuroscientific analysis.

---

## Features

- EEG & DBS Data Loading: Supports common EEG formats (.set, .edf, .fif) and JSON DBS data.
- Peak Detection: Detects the maximum EEG and DBS peaks to align the signals.
- Signal Synchronization: Crops and resamples EEG & DBS data for perfect alignment.
- Visualization: Generates overlay plots of synchronized signals.
- Data Saving: Allows saving synchronized data in `.fif` (EEG) and `.csv` (DBS).
- Customizable Parameters: Adjustable frequency ranges, decimation factors, and peak detection settings.

---

## Installation

1. Clone the Repository:
   ```bash
   git clone https://github.com/lenasalz/dbs-eeg-sync.git
   cd dbs-eeg-sync
   ```

2. Create a Virtual Environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate.bat  # Windows
   ```

3. Install Dependencies with UV:
   ```bash
   uv pip install -r requirements.txt
   ```

**Dependencies include:**
- `numpy`, `pandas`, `mne`, `matplotlib`, `scipy`

---

## Project Structure

```
.
├── README.md
├── config/                # Configuration files
├── data/                  # Raw and synchronized EEG & DBS data
│   ├── eeg_raw.set
│   ├── report.json
│   ├── synchronized_dbs.csv
│   └── synchronized_eeg.fif
├── docs/                  # Documentation
├── examples/              # Example files and scripts
├── notebooks/             # Jupyter Notebooks for analysis
│   └── synchronizer.ipynb
├── plots/                 # Generated plots for peak detection & synchronization
│   ├── eeg_dbs_overlay.png
│   ├── syncPeakDBS.png
│   └── syncPeakEEG.png
├── source/                # Source code for the EEG-DBS synchronization
│   ├── data_loader.py
│   ├── main.py
│   ├── synchronizer.py
│   └── utils/
├── tests/                 # Unit tests
│   └── test_data_loader.py
├── pyproject.toml         # Project configuration
├── uv.lock                # Lockfile for UV dependency management
└── sync_log.txt           # Log file with detected peaks
```

---

## Usage Instructions

Run the Main Script:
```bash
python source/main.py
```

### Step-by-Step Workflow

1. Enter EEG File Path: Path to EEG file (e.g., `.set` file).
2. Enter DBS File Path: Path to DBS JSON report.
3. Peak Detection: Automatic detection of prominent peaks from reduction of deep-brain stimulation. 
4. Synchronization Prompt: Confirm synchronization.
5. Save Data: Optionally save the synchronized EEG & DBS data.

### Command-Line Arguments (Optional)
```bash
python source/main.py --eeg path/to/eeg_raw.set --dbs path/to/dbs.json
```

---

## Example Workflow

Sample Run:
```bash
Enter EEG file path: data/eeg_raw.set
Enter DBS file path: data/Report_Json_Session_Report.json
Peak detected at 23116 samples (11.56s)
DBS peak detected at 15023 samples (5.6s)
Synchronize EEG and DBS? (yes/no): yes
Save synchronized EEG & DBS data? (yes/no): yes
EEG and DBS synchronized and saved successfully.
```

---

## Visualization Examples

### Peak Detection
- EEG Peak Detection: `plots/syncPSD.png`
- DBS Peak Detection: `plots/dbs_peak.png`

### Synchronized Signals Overlay
- EEG & DBS Overlay: `plots/eeg_dbs_overlay.png`

---

## File Outputs

After successful synchronization, the following files are saved:

- EEG Data: `synchronized_data/synchronized_eeg.fif`
- DBS Data: `synchronized_data/synchronized_dbs.csv`
- Logs: `sync_log.txt`

---

## Troubleshooting

1. Mismatched Signal Lengths
- Error: `ValueError: x and y must have same first dimension`
- Fix: Ensure EEG & DBS data are correctly resampled and trimmed.

2. Missing Files
- Error: `FileNotFoundError`
- Fix: Check the file paths and filenames.

3. Plot Not Showing
- Fix: Ensure `matplotlib` is installed and `plt.show()` is called.

---

## Contributing

1. Fork the repository.
2. Create a new branch.
3. Submit a Pull Request with detailed description.

---

## License

This project is licensed under the MIT License.

---

## References
- MNE-Python Documentation: [https://mne.tools/stable/](https://mne.tools/stable/)
- Scipy Signal Processing: [https://docs.scipy.org/doc/scipy/reference/signal.html](https://docs.scipy.org/doc/scipy/reference/signal.html)
- EEG Analysis Resources: OpenNeuro, EEG-BIDS Guidelines

---

Happy Syncing!

