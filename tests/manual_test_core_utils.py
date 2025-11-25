from pathlib import Path
from dbs_eeg_sync.core import ensure_matplotlib_headless, ensure_output_dir, sync_run

# Force headless backend (should log a DEBUG line if --verbose is on in the caller)
ensure_matplotlib_headless(True)

# Ensure an output directory (should log a DEBUG line)
ensure_output_dir(Path("outputs"))

# Run the pipeline with the example data
res = sync_run(
    sub_id="example",
    block="example",
    eeg_file=Path("data/eeg_example.set"),
    dbs_file=Path("data/dbs_example.json"),
    time_range=None,
)

print("EEG pick (idx, s):", res.eeg_sync_idx, res.eeg_sync_s)
print("DBS pick (idx, s):", res.dbs_sync_idx, res.dbs_sync_s)
print("EEG fs / DBS fs:", res.eeg_fs, res.dbs_fs)