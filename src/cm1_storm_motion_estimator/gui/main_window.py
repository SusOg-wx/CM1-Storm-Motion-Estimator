from __future__ import annotations

from PySide6 import QtWidgets


class CM1MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CM1 Storm Motion Estimator")
        self.resize(1400, 900)
        self._build_ui()

    def _build_ui(self) -> None:
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.header = QtWidgets.QLabel("CM1 Storm Motion Estimator")
        self.header.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.main_layout.addWidget(self.header)

        self.tabs = QtWidgets.QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.sim_tab = QtWidgets.QWidget()
        self.sim_layout = QtWidgets.QVBoxLayout(self.sim_tab)
        self.sim_layout.addWidget(QtWidgets.QLabel("Simulation Manager"))
        self.sim_layout.addWidget(QtWidgets.QPushButton("Load Simulation"))
        self.sim_layout.addWidget(QtWidgets.QPushButton("Add Second Simulation"))

        self.analysis_tab = QtWidgets.QWidget()
        self.analysis_layout = QtWidgets.QVBoxLayout(self.analysis_tab)
        self.analysis_layout.addWidget(QtWidgets.QLabel("Tracking Settings"))
        self.analysis_layout.addWidget(QtWidgets.QPushButton("Run Analysis"))

        self.tabs.addTab(self.sim_tab, "Simulation")
        self.tabs.addTab(self.analysis_tab, "Tracking")
