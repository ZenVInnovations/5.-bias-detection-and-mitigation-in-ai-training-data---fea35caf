import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QTableWidget, QTableWidgetItem, QTabWidget, QComboBox, QTextEdit, QHBoxLayout, QGroupBox,
    QStatusBar
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from bias_detection import detect_bias
from bias_mitigation import oversample_minority, undersample_majority

class BiasDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bias Detection and Mitigation in AI Training Data")
        self.setGeometry(100, 100, 900, 700)
        self.setWindowIcon(QIcon())  # You can set a custom icon here

        self.data = None
        self.bias_results = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Data Tab
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Data")
        self.data_layout = QVBoxLayout()
        self.data_tab.setLayout(self.data_layout)

        self.load_button = QPushButton("Load Dataset")
        self.load_button.setFont(QFont("Segoe UI", 10))
        self.load_button.setStyleSheet("padding: 8px;")
        self.load_button.clicked.connect(self.load_dataset)
        self.data_layout.addWidget(self.load_button)

        self.data_preview = QTableWidget()
        self.data_preview.setAlternatingRowColors(True)
        self.data_preview.setStyleSheet("alternate-background-color: #f0f0f0; background-color: #ffffff;")
        self.data_layout.addWidget(self.data_preview)

        # Bias Detection Tab
        self.bias_tab = QWidget()
        self.tabs.addTab(self.bias_tab, "Bias Detection")
        self.bias_layout = QVBoxLayout()
        self.bias_tab.setLayout(self.bias_layout)

        self.sensitive_attr_group = QGroupBox("Select Sensitive Attributes")
        self.sensitive_attr_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.sensitive_attr_layout = QHBoxLayout()
        self.sensitive_attr_group.setLayout(self.sensitive_attr_layout)
        self.bias_layout.addWidget(self.sensitive_attr_group)

        self.sensitive_attr_combo = QComboBox()
        self.sensitive_attr_combo.setEditable(True)
        self.sensitive_attr_combo.setFont(QFont("Segoe UI", 10))
        self.sensitive_attr_layout.addWidget(QLabel("Sensitive Attribute:"))
        self.sensitive_attr_layout.addWidget(self.sensitive_attr_combo)

        self.target_attr_group = QGroupBox("Select Target Attribute")
        self.target_attr_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.target_attr_layout = QHBoxLayout()
        self.target_attr_group.setLayout(self.target_attr_layout)
        self.bias_layout.addWidget(self.target_attr_group)

        self.target_attr_combo = QComboBox()
        self.target_attr_combo.setEditable(True)
        self.target_attr_combo.setFont(QFont("Segoe UI", 10))
        self.target_attr_layout.addWidget(QLabel("Target Attribute:"))
        self.target_attr_layout.addWidget(self.target_attr_combo)

        self.detect_button = QPushButton("Detect Bias")
        self.detect_button.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.detect_button.setStyleSheet("background-color: #007ACC; color: white; padding: 10px; border-radius: 5px;")
        self.detect_button.clicked.connect(self.detect_bias)
        self.detect_button.setEnabled(False)
        self.bias_layout.addWidget(self.detect_button)

        # Bias Mitigation Tab
        self.mitigation_tab = QWidget()
        self.tabs.addTab(self.mitigation_tab, "Bias Mitigation")
        self.mitigation_layout = QVBoxLayout()
        self.mitigation_tab.setLayout(self.mitigation_layout)

        self.mitigate_button = QPushButton("Mitigate Bias (Oversample Minority)")
        self.mitigate_button.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.mitigate_button.setStyleSheet("background-color: #28A745; color: white; padding: 10px; border-radius: 5px;")
        self.mitigate_button.clicked.connect(self.mitigate_bias)
        self.mitigate_button.setEnabled(False)
        self.mitigation_layout.addWidget(self.mitigate_button)

        self.export_button = QPushButton("Export Dataset")
        self.export_button.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.export_button.setStyleSheet("background-color: #6C757D; color: white; padding: 10px; border-radius: 5px;")
        self.export_button.clicked.connect(self.export_data)
        self.export_button.setEnabled(False)
        self.mitigation_layout.addWidget(self.export_button)

        # Log Output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 10))
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; padding: 10px;")
        self.main_layout.addWidget(self.log_output)

        # Matplotlib Figure for plots
        self.figure = plt.figure(facecolor="#f9f9f9")
        self.canvas = FigureCanvas(self.figure)
        self.bias_layout.addWidget(self.canvas)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def log_message(self, message):
        self.log_output.append(message)
        self.status_bar.showMessage(message, 5000)  # Show message for 5 seconds

    def load_dataset(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)", options=options)
        if file_name:
            try:
                if file_name.endswith('.csv'):
                    self.data = pd.read_csv(file_name)
                else:
                    self.data = pd.read_excel(file_name)
                self.log_message(f"Loaded dataset: {file_name}")
                self.show_data_preview()
                self.populate_attribute_combos()
                self.detect_button.setEnabled(True)
                self.mitigate_button.setEnabled(False)
                self.export_button.setEnabled(False)
            except Exception as e:
                self.log_message(f"Failed to load dataset: {e}")

    def show_data_preview(self):
        if self.data is not None:
            self.data_preview.clear()
            self.data_preview.setRowCount(min(len(self.data), 10))
            self.data_preview.setColumnCount(len(self.data.columns))
            self.data_preview.setHorizontalHeaderLabels(self.data.columns)
            for i in range(min(len(self.data), 10)):
                for j, col in enumerate(self.data.columns):
                    item = QTableWidgetItem(str(self.data.iloc[i, j]))
                    self.data_preview.setItem(i, j, item)

    def populate_attribute_combos(self):
        if self.data is not None:
            columns = list(self.data.columns)
            self.sensitive_attr_combo.clear()
            self.target_attr_combo.clear()
            self.sensitive_attr_combo.addItems(columns)
            self.target_attr_combo.addItems(columns)
            # Set default selections if possible
            for default in ['gender', 'race', 'ethnicity']:
                if default in columns:
                    self.sensitive_attr_combo.setCurrentText(default)
                    break
            if 'target' in columns:
                self.target_attr_combo.setCurrentText('target')
            else:
                self.target_attr_combo.setCurrentIndex(0)

    def detect_bias(self):
        sensitive_attr = self.sensitive_attr_combo.currentText()
        target_attr = self.target_attr_combo.currentText()

        if self.data is None:
            self.log_message("No dataset loaded.")
            return
        if sensitive_attr not in self.data.columns or target_attr not in self.data.columns:
            self.log_message("Selected attributes not in dataset.")
            return

        try:
            results = detect_bias(self.data, [sensitive_attr], target_attr)
            self.bias_results = results
            self.log_message(f"Bias detection results for attribute '{sensitive_attr}':")
            self.log_message(str(results[sensitive_attr]))

            # Plot distribution
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            dist = results[sensitive_attr]['distribution']
            ax.bar(dist.keys(), dist.values(), color='#007ACC')
            ax.set_title(f"Distribution of {sensitive_attr}", fontsize=12, fontweight='bold')
            ax.set_ylabel("Proportion", fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            self.canvas.draw()

            self.mitigate_button.setEnabled(True)
            self.export_button.setEnabled(False)
        except Exception as e:
            self.log_message(f"Error during bias detection: {e}")
            self.mitigate_button.setEnabled(False)

    def mitigate_bias(self):
        if self.data is None or self.bias_results is None:
            self.log_message("No data or bias detection results available.")
            return

        target_attr = self.target_attr_combo.currentText()
        sensitive_attr = self.sensitive_attr_combo.currentText()

        class_counts = self.data[target_attr].value_counts()
        if len(class_counts) < 2:
            self.log_message("Not enough classes for mitigation.")
            self.export_button.setEnabled(False)
            return

        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        try:
            self.data = oversample_minority(self.data, target_attr, minority_class)
            self.log_message(f"Bias mitigation applied: minority class '{minority_class}' oversampled.")
            self.show_data_preview()
            self.export_button.setEnabled(True)
        except Exception as e:
            self.log_message(f"Error during bias mitigation: {e}")
            self.export_button.setEnabled(False)

    def export_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Dataset", "", "CSV Files (*.csv)", options=options)
        if file_name:
            try:
                self.data.to_csv(file_name, index=False)
                self.log_message(f"Dataset exported to: {file_name}")
            except Exception as e:
                self.log_message(f"Failed to export dataset: {e}")
