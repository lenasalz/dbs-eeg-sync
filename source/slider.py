import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QLabel, QPushButton
from PyQt5.QtCore import Qt
import datetime
import csv

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class PowerSliderApp(QWidget):
    def __init__(self, power_data, sub_id, block):
        super().__init__()
        self.setWindowTitle("EEG Power Sample Selector")

        self.sub_id = sub_id
        self.block = block

        # Load data
        self.power_data = power_data
        self.selected_index = 0

        # UI
        layout = QVBoxLayout()

        self.label = QLabel(f"Selected Index: {self.selected_index}")
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.power_data) - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_index)
        layout.addWidget(self.slider)

        # Plot
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Save & Quit Button
        self.save_button = QPushButton("Save and Quit")
        self.save_button.clicked.connect(self.save_and_quit)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.plot_data()

    def save_to_csv(self):
        output_path = "selected_indices.csv"
        timestamp = datetime.datetime.now().isoformat()
        header = ["timestamp", "sub_id", "block", "sync_index"]

        # Check if file exists
        try:
            with open(output_path, "x", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
        except FileExistsError:
            pass

        # Append the final entry
        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, self.sub_id, self.block, self.selected_index])

    def save_and_quit(self):
        self.save_to_csv()
        self.close()

    def closeEvent(self, event):
        self.save_to_csv()
        event.accept()

    def update_index(self, value):
        self.selected_index = value
        self.label.setText(f"Selected Index: {self.selected_index}")
        self.plot_data()

    def plot_data(self):
        self.ax.clear()
        self.ax.plot(self.power_data.index, self.power_data.iloc[:, 0], label="Power")
        self.ax.axvline(x=self.selected_index, color='red', linestyle='--', label='Selected Index')
        self.ax.set_title("EEG Power Data")
        self.ax.set_xlabel("Sample Index")
        self.ax.set_ylabel("Power")
        self.ax.legend()
        self.canvas.draw()


def run_manual_sync_slider(csv_path, sub_id, block):
    """
    Launch the EEG power selection slider GUI.
    This can be called from other scripts after user declines auto sync.
    """
    app = QApplication(sys.argv)
    window = PowerSliderApp(csv_path, sub_id, block)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_manual_sync_slider(
        "/Users/lenasalzmann/dev/dbs-eeg-sync/data/P4-2004_pre8walk_eeg_power.csv",
        sub_id="P4-2004",
        block="pre8walk"
    )