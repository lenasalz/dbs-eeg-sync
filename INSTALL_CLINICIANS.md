# Quick start for clinicians (no command line)

This guide is for users who just want to **synchronize an EEG recording with a
DBS recording** and are not comfortable with the command line. It uses a simple
window where you pick your files and press one button.

If you are a developer, see the main [`README.md`](README.md) instead.

---

## Step 1 — Install Python (one time only)

The toolbox runs on Python. You only need to do this once per computer.

1. Go to <https://www.python.org/downloads/>.
2. Download **Python 3.12 or newer** for your operating system.
3. Run the installer.
   - **Windows:** on the first installer screen, tick the box
     **"Add Python to PATH"**, then click *Install Now*.
   - **macOS:** just run the downloaded `.pkg` and follow the prompts.

> Not sure if you already have it? You can skip ahead to Step 2 — the launcher
> will tell you if Python is missing.

---

## Step 2 — Get the toolbox folder

If someone sent you a `.zip` of the project, unzip it to a folder you can find
again (for example your Desktop or Documents). You should end up with a folder
called `dbs-eeg-sync` containing a `launchers` folder.

---

## Step 3 — Double-click the launcher

Open the `launchers` folder and double-click the file for your system:

| System  | File to double-click                          |
| ------- | --------------------------------------------- |
| macOS   | `Launch dbs-eeg-sync (macOS).command`         |
| Windows | `Launch dbs-eeg-sync (Windows).bat`           |

**The first time only**, the launcher sets up everything it needs. This can
take a few minutes and a black/terminal window will show progress text — that
is normal. After it finishes, the synchronization window opens.

> **macOS security note:** the first time, macOS may refuse to run the file
> ("cannot be opened because it is from an unidentified developer"). If so,
> **right-click** the file → **Open** → **Open**. You only need to do this once.

<!-- TODO: add screenshot of the launcher / first-run terminal here -->
<!-- ![Launcher first run](docs/img/launcher_first_run.png) -->

---

## Step 4 — Synchronize your recordings

In the window that opens:

1. **EEG file** — click *Browse…* and choose your EEG file (`.set`, `.edf`,
   `.fif`, …).
2. **DBS JSON file** — click *Browse…* and choose the `.json` file exported from
   the Percept device.
3. **Output folder** — where results should be saved (defaults to `outputs`).
4. **Subject ID** and **Block / session** — short labels for this recording
   (for example `S01` and `B1`).
5. **Time range** *(optional)* — if you only want to look at part of the
   recording, type `start,end` in seconds, e.g. `10,60`. Leave it blank to use
   the whole recording.
6. Leave **"Pick sync point by hand"** ticked to confirm the alignment visually,
   or untick it to let the tool detect the stimulation artifact automatically.
7. Press **Run synchronization**.

<!-- TODO: add screenshot of the main launcher window here -->
<!-- ![Main window](docs/img/main_window.png) -->

If you left manual selection on, a second window appears with sliders. Drag the
dashed line to the moment the stimulation artifact starts in each trace, then
press **Confirm selection**.

<!-- TODO: add screenshot of the slider / manual-sync window here -->
<!-- ![Manual sync window](docs/img/manual_sync.png) -->

When it finishes, a message tells you where the results were saved. Inside the
output folder you will find:

- a `*_metadata_*.json` file with the synchronization result, and
- a `plots/` folder with the overlay and artifact figures (if you kept
  *Save plots* ticked).

---

## Trying it without your own data

Want to see it work first? You can run the bundled example data from a terminal:

```
dbs-eeg-sync --test --plots --headless
```

(Or ask a colleague who uses the command line to show you once.)

---

## If something goes wrong

- **"Python was not found"** — install Python (Step 1), and on Windows make
  sure you ticked *Add Python to PATH*, then try the launcher again.
- **The window does not open / mentions PyQt6** — the launcher tries to install
  the graphical components automatically; if it cannot, ask a technical
  colleague to run `pip install "dbs-eeg-sync[gui]"` in the project folder.
- **"Synchronization could not be completed"** — double-check that the EEG and
  DBS files belong to the same recording and that any time range you typed is
  inside the recording.

For anything else, contact the maintainer listed in the main
[`README.md`](README.md).
