# EEG-DBS Synchronization Pipeline

**EEG-DBS Synchronization Pipeline** is a Python-based tool designed to synchronize EEG (Electroencephalography) and DBS (Deep Brain Stimulation) BrainSense:tm: data. It enables efficient data loading, peak detection, alignment, and visualization of EEG and DBS signals for neuroscientific analysis.

## TEMPORARY NOTES ## 
- The drop in the EEG signal produced by the reduction in DBS stimulation amplitude is not the same in all EEG channels. Occipial and medial channels appear to have a clearer change (drop) in signal. 
- Sometimes, there is a peak shortly after the drop. Is it noise, or some neurophysiological artefact related to the reduction in amplitude?
- Potentially try to find the best drop in all channels first, and use this for syncrhonization?-
- left and right electrodes seem sometimes opposed with drop / spike

## Recording Synchronization Procedure
1. Start the EEG recording
2. Start the DBS Streaming 
3. Reduce the DBS amplitude by at least 0.5mA to indcue an artifact in the DBS **and** EEG data.
4. Increase the amplitude again in the BrainSense:tm: App and wait until it is back to the initial stimulation.  

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

1. Clone the repository:
Open a terminal and run:
   ```bash
   git clone https://github.com/lenasalz/dbs-eeg-sync.git
   cd dbs-eeg-sync
   ```

2. Set up the environment using uv:
This project uses **uv** to manage the Python environment and dependencies. Please follow the [official uv docs](https://docs.astral.sh/uv/) for installation. 
    ```bash
    uv sync 
    source .venv/bin/activate 
   ```
   - uv sync will:
    - create a .venv folder (if it doesn't exist)
    - install dependencies from pyproject.toml and/or uv.lock
*If a uv.lock file exists, uv will use it to ensure reproducible installs. Otherwise, it installs from pyproject.toml.*


---

## Project Structure

```
.
├── README.md
├── config
├── data
│   ├── Report_Json_Session_Report_20241025T120701.json
│   ├── eeg_example.set
│   ├── synchronized_dbs.csv
│   └── synchronized_eeg.fif
├── docs
├── examples
├── notebooks
│   └── synchronizer.ipynb
├── plots
│   ├── eeg_dbs_overlay.png
│   ├── syncPeakDBS.png
│   └── syncPeakEEG.png
├── pyproject.toml
├── source
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── data_loader.cpython-312.pyc
│   │   ├── sync_peaks_finder.cpython-312.pyc
│   │   └── synchronizer.cpython-312.pyc
│   ├── data_loader.py
│   ├── main.py
│   ├── sync_peaks_finder.py
│   ├── synchronizer.py
│   └── utils
│       └── __init__.py
├── sync_log.txt
├── tests
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   └── test_data_loader.cpython-312.pyc
│   ├── test_data_loader.py
│   ├── test_sync_peak_finder.py
│   └── test_synch
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
Enter Frequency range of artifact and duration of signals
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
---
