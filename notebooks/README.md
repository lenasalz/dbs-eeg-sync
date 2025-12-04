# Jupyter Notebooks

This directory contains Jupyter notebooks demonstrating the synchronization methods, validation analyses, and figure generation for the manuscript.

## 📚 Notebook Overview

### Core Method Demonstration

#### `sync_artifact_detection.ipynb`
Demonstrates the core artifact detection algorithm for synchronizing EEG and DBS signals. Shows step-by-step how stimulation artifacts are identified in both recording modalities.

**Key outputs:**
- Artifact detection in EEG power spectrum
- Artifact detection in DBS time series
- Visual comparison of detected sync points

#### `sync_drop_detection.ipynb`
Focuses specifically on "drop" type artifacts where stimulation causes a power decrease in the EEG signal. Complements the main artifact detection method.

**Key outputs:**
- Power drop identification
- Threshold selection for drop artifacts
- Validation of drop-based synchronization

---

### Validation & Analysis

#### `validation_metrics.ipynb`
Comprehensive validation of the synchronization method including:
- Comparison with manual synchronization
- Accuracy metrics (temporal precision, consistency)
- Performance across different subjects and recording conditions

**Key outputs:**
- Validation statistics
- Manual vs. automated sync comparison plots
- Accuracy metrics tables

#### `sync_duration_analysis.ipynb`
Analyzes the temporal characteristics and duration of synchronization artifacts.

**Key outputs:**
- Artifact duration distributions
- Timing stability analysis
- Recommendations for detection window sizes

#### `test_sync_methods.ipynb`
Compares different synchronization approaches and parameter choices.

**Key outputs:**
- Method comparison results
- Parameter sensitivity analysis
- Best practice recommendations

---

### Special Cases

#### `double_sync.ipynb`
Handles edge cases where multiple synchronization artifacts occur in the recording, requiring disambiguation between true sync points and subsequent stimulation changes.

**Key outputs:**
- Multi-artifact detection plots
- Logic for selecting the correct sync point
- Validation of edge case handling

#### `epi_sync_test.ipynb`
Specific test case demonstrating synchronization on a particular dataset. Contains figures used in the publication.

**Key outputs:**
- Dataset-specific validation
- Publication-quality figures

---

### Publication Figures

#### `publication_figures.ipynb`
Generates all publication-quality figures for the manuscript. This is the main notebook for reproducing paper figures.

**Key outputs:**
- All manuscript figures in publication format
- High-resolution plots (PDF/SVG)
- Saved to `notebooks/outputs/plots/`

---

## 🚀 Getting Started

### Prerequisites

Install the package with notebook dependencies:

```bash
# Using uv
uv sync
uv pip install -e ".[notebooks]"

# Or using pip
pip install -e ".[notebooks]"
```

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use Jupyter Lab
jupyter lab
```

Navigate to the `notebooks/` directory and open any notebook.

---

## 📂 Output Structure

Notebooks generate outputs in the `notebooks/outputs/` directory:

```
notebooks/outputs/
├── outputData/          # Intermediate data files
└── plots/              # Generated figures
    ├── fig*.pdf        # Publication figures
    ├── fig*.svg        # Vector graphics
    └── v2/            # Figure iterations (if any)
```

---

## 📝 Note on Data

These notebooks expect EEG and DBS data files to be available. Due to data privacy and size constraints, the example data in the `data/` directory is limited. For full reproduction of publication results, contact the authors for access to the complete dataset.

---

## 🤝 Contributing

If you develop new analysis notebooks or improve existing ones, please:
1. Add clear markdown documentation explaining the notebook's purpose
2. Include expected inputs and outputs
3. Test that the notebook runs from start to finish
4. Update this README with a description of your notebook

---

## 📬 Questions?

For questions about specific notebooks or analyses, please:
- Open an issue on [GitHub](https://github.com/lenasalz/dbs-eeg-sync/issues)
- Contact: [lena.salzmann@hest.ethz.ch](mailto:lena.salzmann@hest.ethz.ch)

