import sys
import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QStackedWidget,
    QPushButton,
    QLabel,
    QFrame,
    QHBoxLayout,
    QFileDialog,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLineEdit,
    QCheckBox,
    QFormLayout,
    QProgressBar, # Added for Progress Bar
    QStatusBar,
)
from PyQt6.QtGui import QPalette, QColor, QPixmap, QImage, QFont, QIcon # Added QFont, QIcon
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import QStyle # For standard icons

# Standard library imports
import time
import json
import pprint
import re
from datetime import datetime # For timestamping history

# Attempt to import project-specific modules
# These are crucial for the analysis function
try:
    from PIL import Image
    import numpy as np # Ensure numpy is available for analysis functions
    import cv2 # Ensure cv2 is available for analysis functions

    import config
    import validation
    # Import all required analysis modules from the project
    from ela_analysis import perform_multi_quality_ela
    from feature_detection import extract_multi_detector_features
    from copy_move_detection import detect_copy_move_advanced, detect_copy_move_blocks, kmeans_tampering_localization
    from advanced_analysis import (analyze_noise_consistency, analyze_frequency_domain,
                                  analyze_texture_consistency, analyze_edge_consistency,
                                  analyze_illumination_consistency, perform_statistical_analysis)
    from jpeg_analysis import advanced_jpeg_analysis, jpeg_ghost_analysis
    from classification import classify_manipulation_advanced #Removed: prepare_feature_vector
    # from visualization import visualize_results_advanced, export_kmeans_visualization # Not directly used by analysis logic, but by main.py CLI
    # from export_utils import export_complete_package # Not directly used by analysis logic
except ImportError as e:
    print(f"Critical Error importing project modules for analysis: {e}. Analysis will likely fail.")
    # Define dummies if imports fail, so GUI can still load
    Image = None
    np = None
    cv2 = None
    import config
    import validation
except ImportError as e:
    print(f"Error importing project modules: {e}. Features may be limited.")
    config = None
    validation = None
    # Dummy definitions for analysis functions if imports fail
    perform_multi_quality_ela = None
    extract_multi_detector_features = None
    detect_copy_move_advanced = None
    detect_copy_move_blocks = None
    kmeans_tampering_localization = None
    analyze_noise_consistency = None
    analyze_frequency_domain = None
    analyze_texture_consistency = None
    analyze_edge_consistency = None
    analyze_illumination_consistency = None
    perform_statistical_analysis = None
    advanced_jpeg_analysis = None
    jpeg_ghost_analysis = None
    classify_manipulation_advanced = None
    # advanced_tampering_localization is defined below if needed by analyze_image_for_gui

# Attempt to import database module
try:
    import database
except ImportError as e_db:
    print(f"Database module import error: {e_db}. History feature will be disabled.")
    database = None

# Attempt to import export_utils functions
EXPORT_UTILS_AVAILABLE = False
export_visualization_png_util = None
export_to_advanced_docx_util = None
export_report_pdf_util = None
export_complete_package_util = None
try:
    from export_utils import (
        export_visualization_png as export_visualization_png_util,
        export_to_advanced_docx as export_to_advanced_docx_util,
        export_report_pdf as export_report_pdf_util,
        export_complete_package as export_complete_package_util
    )
    EXPORT_UTILS_AVAILABLE = True
    print("Export utilities imported successfully.")
except ImportError as e_export:
    print(f"Export_utils import error: {e_export}. Export feature will be limited/disabled.")


# Attempt to import Matplotlib and related visualization tools
MATPLOTLIB_AVAILABLE = False
FigureCanvas = None
Figure = None
gridspec = None
# PdfPages = None # Not strictly needed for GUI display but part of visualization.py
sobel = None # from skimage.filters, used in visualization

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib imported successfully for GUI.")
except ImportError:
    print("Matplotlib (or backend_qtagg) not found. Detailed visualization will be disabled.")

try:
    from skimage.filters import sobel
except ImportError:
    print("scikit-image (skimage.filters.sobel) not found. Some visualizations may be affected.")


MAX_PROCESSING_PAGES = 25 # Max number of pages for individual processing steps

# Custom Exceptions for GUI validation flow
class InvalidFileType(Exception):
    pass

class FileSizeError(Exception):
    pass

class InterruptedException(Exception): # Custom exception for cancellation
    pass

# Analysis Worker
class AnalysisWorker(QObject):
    step_started_signal = pyqtSignal(int, str)
    step_completed_signal = pyqtSignal(int, str, object)
    analysis_finished_signal = pyqtSignal(object)
    analysis_error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_running = False
        self._is_cancelled = False # For cancellation

    def run_analysis(self, image_path, output_dir):
        self.is_running = True
        self._is_cancelled = False # Reset cancel flag on new run
        try:
            if not all([Image, np, cv2, config, validation, perform_multi_quality_ela]):
                raise ImportError("One or more core analysis modules are not available.")

            results = analyze_image_for_gui(image_path, output_dir, self)
            if self.is_running and not self._is_cancelled:
                self.analysis_finished_signal.emit(results)
            elif self._is_cancelled:
                 # If _is_cancelled is true but InterruptedException wasn't raised from analyze_image_for_gui
                 # (e.g. cancelled right at the end), ensure we signal cancellation.
                 # This path might not be strictly necessary if checks in analyze_image_for_gui are thorough.
                self.analysis_error_signal.emit("Analysis Cancelled by User")

        except InterruptedException: # Catch explicit cancellation
            if self.is_running: # ensure is_running is true before emitting error for cancellation
                 self.analysis_error_signal.emit("Analysis Cancelled by User")
        except Exception as e:
            if self.is_running: # Avoid emitting error if stop_analysis was called AND error occurred
                import traceback
                self.analysis_error_signal.emit(f"{str(e)}\n{traceback.format_exc()}")
        finally:
            self.is_running = False
            self._is_cancelled = False # Reset flag

    def cancel(self):
        print("Worker: Cancel requested.")
        self._is_cancelled = True
        # self.is_running = False # Optionally, setting is_running to False can also help break loops in analyze_image_for_gui

    def stop_analysis(self): # Kept for compatibility, but cancel is more explicit
        print("Worker: Stop analysis called (similar to cancel).")
        self.cancel()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.InvalidFileType = InvalidFileType
        self.FileSizeError = FileSizeError

        self.setWindowTitle("Advanced Forensic Image Analysis System GUI")
        self.setGeometry(100, 100, 1000, 800)
        self.current_image_path = None
        self.current_pil_image = None
        self.current_analysis_results = None
        self.output_dir = None

        # Apply a global stylesheet for consistency
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1; /* Light gray for window background */
            }
            QPushButton {
                background-color: #3498db; /* Primary blue */
                color: white;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #2980b9; /* Darker blue border */
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4fa8df; /* Lighter blue for hover */
            }
            QPushButton:disabled {
                background-color: #a6acaf;
                color: #e5e7e9;
                border-color: #95a5a6;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #bdc3c7; /* Mid-gray border */
                padding: 5px;
                background-color: white;
                color: #333333; /* Dark gray text */
                border-radius: 3px;
                font-size: 13px;
            }
            QLabel {
                color: #333333; /* Dark gray text for labels */
                font-size: 13px;
            }
            QCheckBox {
                font-size: 13px;
                spacing: 5px; /* Space between checkbox and text */
            }
            QTableWidget {
                border: 1px solid #bdc3c7;
                alternate-background-color: #f8f8f8; /* Slightly lighter than #f2f2f2 */
                selection-background-color: #3498db; /* Blue for selection */
                gridline-color: #ddeeff; /* Light blue grid lines */
            }
            QHeaderView::section {
                background-color: #ddeeff; /* Light blue for headers */
                padding: 6px;
                border: 1px solid #cccccc;
                font-size: 13px;
                font-weight: bold;
            }
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                text-align: center;
                background-color: #e0e0e0; /* Light gray background for bar */
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #3498db; /* Blue for progress chunk */
                width: 1px; /* Required for chunk to show */
                margin: 1px;
                border-radius: 3px;
            }
            QStatusBar {
                background-color: #ddeeff; /* Light blue for status bar */
                font-size: 12px;
            }
            QStatusBar QLabel { /* For labels directly in status bar */
                color: #2c3e50; /* Darker blue-gray for status text */
                font-size: 12px;
            }
        """)


        # Initialize database
        if database:
            database.init_db()

        # Matplotlib setup for Results page
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        if self.matplotlib_available:
            self.results_figure = Figure(figsize=(28, 20))
            self.results_canvas = FigureCanvas(self.results_figure)
        else:
            self.results_figure = None
            self.results_canvas = QLabel("Matplotlib is not available. Detailed visualization cannot be displayed.")
            self.results_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Style will be picked from global QLabel, but can override
            self.results_canvas.setStyleSheet("font-size: 14px; color: #e74c3c; border: 1px solid #ccc; padding: 10px; background-color: #fff0f0;")

        # Analysis Thread Setup
        self.analysis_thread = QThread()
        self.analysis_worker = AnalysisWorker()
        self.analysis_worker.moveToThread(self.analysis_thread)

        # Connect worker signals to MainWindow slots
        self.analysis_worker.step_started_signal.connect(self.on_step_started)
        self.analysis_worker.step_completed_signal.connect(self.on_step_completed)
        self.analysis_worker.analysis_finished_signal.connect(self.on_analysis_finished)
        self.analysis_worker.analysis_error_signal.connect(self.on_analysis_error)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0) # Remove margins for the main layout
        self.main_layout.setSpacing(0) # Remove spacing for the main layout


        # Header/Navigation Bar
        self.header_bar = QFrame()
        self.header_bar.setStyleSheet("background-color: #2980b9; color: white;") # Darker blue for header
        self.header_bar.setFixedHeight(50)
        self.main_layout.addWidget(self.header_bar)

        header_layout = QHBoxLayout(self.header_bar)
        header_layout.setContentsMargins(10, 0, 10, 0) # Add some horizontal margin to header content

        # Navigation Buttons in Header
        self.nav_buttons = {}
        button_names = ["Upload", "Processing", "Results", "History", "Export"]
        for name in button_names:
            button = QPushButton(f"Go to {name}")
            # Specific style for header buttons to stand out or blend as desired
            button.setStyleSheet("""
                QPushButton {
                    background-color: #3498db; border: 1px solid #2980b9;
                    padding: 8px 12px; font-size: 13px;
                }
                QPushButton:hover { background-color: #4fa8df; }
                QPushButton:disabled { background-color: #7f8c8d; border-color: #707b7c; }
            """)
            button.clicked.connect(lambda checked, n=name: self.switch_page(n))
            self.nav_buttons[name] = button
            header_layout.addWidget(button)

        header_layout.addStretch()


        # StackedWidget for pages
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget, 1)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar) # Set it as the actual QMainWindow status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 18)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar, 1)

        self.cancel_analysis_button = QPushButton("Cancel") # Shorter text
        self.cancel_analysis_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton)) # Icon
        self.cancel_analysis_button.clicked.connect(self.request_cancel_analysis)
        self.cancel_analysis_button.setEnabled(False)
        self.cancel_analysis_button.setStyleSheet("""
            QPushButton { background-color: #e74c3c; border-color: #c0392b; padding: 4px 8px; font-size: 12px;}
            QPushButton:hover { background-color: #ec7063; }
            QPushButton:disabled { background-color: #a6acaf; border-color: #95a5a6; }
        """)
        self.status_bar.addPermanentWidget(self.cancel_analysis_button)

        self.global_status_label = QLabel("Ready.")
        self.status_bar.addWidget(self.global_status_label, 1)


        # Pages (Main static pages like Upload, Results, etc.)
        self.pages = {}
        page_names = ["Upload", "Results", "History", "Export"]

        for name in page_names:
            widget = QWidget()
            # Page background will be set by QMainWindow's stylesheet or can be overridden here
            # widget.setStyleSheet("background-color: #ecf0f1;")

            if name == "Upload":
                self.setup_upload_page(widget)
            elif name == "Results":
                self.setup_results_page(widget)
            elif name == "History":
                self.setup_history_page(widget)
            elif name == "Export":
                self.setup_export_page(widget)
            else:
                temp_layout = QVBoxLayout(widget)
                temp_label = QLabel(f"This is the {name} Page (Unconfigured)")
                temp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                temp_layout.addWidget(temp_label)

            self.stacked_widget.addWidget(widget)
            self.pages[name] = widget

        # Create individual processing step pages
        self.processing_pages = {}
        for i in range(MAX_PROCESSING_PAGES): # Create a fixed number of pages
            page = QWidget()
            page.setStyleSheet("background-color: #fafafa;") # Slightly different background for processing steps
            layout = QVBoxLayout(page)

            step_title_label = QLabel(f"Step {i}: Waiting for analysis to start...")
            step_title_label.setObjectName("step_title_label") # For easy finding
            step_title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px; color: #2c3e50;")
            layout.addWidget(step_title_label)

            image_display_label = QLabel()
            image_display_label.setObjectName("image_display_label")
            image_display_label.setMinimumSize(300, 200)
            image_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_display_label.setStyleSheet("border: 1px solid #bdc3c7; margin-top: 5px;")
            image_display_label.hide() # Initially hidden
            layout.addWidget(image_display_label)

            text_display_area = QTextEdit()
            text_display_area.setObjectName("text_display_area")
            text_display_area.setReadOnly(True)
            text_display_area.setStyleSheet("background-color: white; border: 1px solid #bdc3c7; font-family: monospace; margin-top: 5px;")
            text_display_area.hide() # Initially hidden
            layout.addWidget(text_display_area)

            layout.setStretchFactor(image_display_label, 1) # Allow image to take space
            layout.setStretchFactor(text_display_area, 1) # Allow text area to take space

            self.processing_pages[i] = page
            self.stacked_widget.addWidget(page)

        # Add a generic "Processing" placeholder page for the nav button if needed,
        # or handle the "Processing" nav button click differently.
        # For now, let's add one that instructs user.
        processing_landing_page = QWidget()
        processing_landing_page.setStyleSheet("background-color: #ecf0f1;")
        plp_layout = QVBoxLayout(processing_landing_page)
        plp_label = QLabel("Start an analysis from the Upload page to see processing steps.")
        plp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plp_layout.addWidget(plp_label)
        self.stacked_widget.addWidget(processing_landing_page)
        self.pages["Processing"] = processing_landing_page # This is for the top nav button "Processing"


        # Set initial page (Upload)
        if "Upload" in self.pages:
            self.switch_page("Upload")


    def setup_upload_page(self, page):
        layout = QVBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15) # Consistent padding
        page.setLayout(layout)

        # Browse Button
        self.browse_button = QPushButton("Browse Image")
        self.browse_button.setStyleSheet(
            "background-color: #2980b9; color: white; padding: 10px; border-radius: 5px; font-size: 16px;"
        )
        self.browse_button.clicked.connect(self.browse_image)
        layout.addWidget(self.browse_button, 0, Qt.AlignmentFlag.AlignHCenter) # Centered horizontally

        # Image Preview Label
        self.image_preview_label = QLabel("Image Preview Area\n\n(Select an image to see preview)")
        self.image_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview_label.setMinimumSize(400, 300)
        self.image_preview_label.setStyleSheet(
            "border: 2px dashed #bdc3c7; color: #7f8c8d; font-size: 14px; margin-top: 10px;"
        )
        layout.addWidget(self.image_preview_label, 1)

        # Validation Status Label
        self.validation_status_label = QLabel("Validation status: N/A")
        self.validation_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.validation_status_label.setStyleSheet("font-size: 14px; margin-top: 5px; padding: 5px;")
        layout.addWidget(self.validation_status_label)

        # Start Analysis Button
        self.start_analysis_button = QPushButton("Start Analysis")
        self.start_analysis_button.setStyleSheet(
            "background-color: #27ae60; color: white; padding: 10px; border-radius: 5px; font-size: 16px;"
        )
        self.start_analysis_button.clicked.connect(self.start_image_analysis)
        self.start_analysis_button.setEnabled(False)
        layout.addWidget(self.start_analysis_button, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(10) # Add some space at the bottom

        layout.setStretchFactor(self.image_preview_label, 1) # Ensure preview area can expand

    def setup_results_page(self, page_widget):
        results_layout = QVBoxLayout(page_widget)
        results_layout.setContentsMargins(15, 15, 15, 15) # Consistent padding

        self.results_summary_text = QTextEdit()
        self.results_summary_text.setReadOnly(True)
        self.results_summary_text.setStyleSheet("font-size: 13px; background-color: #fdfefe; border: 1px solid #ccc;")
        self.results_summary_text.setMinimumHeight(150) # Ensure enough space for text
        results_layout.addWidget(self.results_summary_text, 1) # Weight for text area

        # Add the Matplotlib canvas (or placeholder label)
        results_layout.addWidget(self.results_canvas, 4)
        page_widget.setLayout(results_layout)

    def setup_history_page(self, page_widget):
        history_layout = QVBoxLayout(page_widget)
        history_layout.setContentsMargins(15, 15, 15, 15) # Consistent padding

        self.refresh_history_button = QPushButton("Refresh History")
        self.refresh_history_button.setStyleSheet(
            "background-color: #3498db; color: white; padding: 8px; border-radius: 3px; margin-bottom: 5px;"
        )
        if database: # Only connect if database module is available
            self.refresh_history_button.clicked.connect(self.load_analysis_history)
        else:
            self.refresh_history_button.setEnabled(False)
            self.refresh_history_button.setText("Refresh History (DB N/A)")

        history_layout.addWidget(self.refresh_history_button)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(7) # Timestamp, Name, Path, Class, Confidence, OutputDir, Placeholder for action
        self.history_table.setHorizontalHeaderLabels([
            "Timestamp", "Image Name", "Classification",
            "Confidence", "Copy-Move Score", "Splicing Score", "Output Directory"
        ])
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # Timestamp
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive) # Name
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents) # Class
        self.history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents) # Confidence
        self.history_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch) # Output Dir
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setStyleSheet("alternate-background-color: #f2f2f2; background-color: white;")

        history_layout.addWidget(self.history_table)
        page_widget.setLayout(history_layout)

        if database:
            self.load_analysis_history()

    def setup_export_page(self, page_widget):
        main_layout = QVBoxLayout(page_widget)
        main_layout.setContentsMargins(15, 15, 15, 15) # Consistent padding
        form_layout = QFormLayout()

        # Directory selection
        dir_layout = QHBoxLayout()
        self.export_dir_edit = QLineEdit("./gui_exports")
        dir_layout.addWidget(self.export_dir_edit)
        browse_dir_button = QPushButton("") # Text removed, icon only
        browse_dir_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        browse_dir_button.setStyleSheet("padding: 4px; width: 40px;") # Make icon button more compact
        browse_dir_button.clicked.connect(self.browse_export_directory)
        dir_layout.addWidget(browse_dir_button)
        form_layout.addRow("Export Directory:", dir_layout)

        # Base name
        self.export_base_name_edit = QLineEdit("analysis_export")
        form_layout.addRow("Base Filename:", self.export_base_name_edit)

        main_layout.addLayout(form_layout)
        main_layout.addSpacing(15)

        # Checkboxes for export options
        self.cb_export_jpg = QCheckBox("Export Main Visualization (JPG from PNG)")
        self.cb_export_docx = QCheckBox("Export Full Report (DOCX)")
        self.cb_export_pdf_report = QCheckBox("Export Full Report (PDF from DOCX)")
        self.cb_export_all = QCheckBox("Export All (Comprehensive Package)")

        # Optional: Link "Export All" to other checkboxes
        # self.cb_export_all.stateChanged.connect(self._toggle_all_exports)

        main_layout.addWidget(QLabel("Select Export Options:"))
        main_layout.addWidget(self.cb_export_jpg)
        main_layout.addWidget(self.cb_export_docx)
        main_layout.addWidget(self.cb_export_pdf_report)
        main_layout.addWidget(self.cb_export_all)
        main_layout.addSpacing(15)

        # Action button
        self.start_export_button = QPushButton("Start Export")
        self.start_export_button.setStyleSheet(
            "background-color: #16a085; color: white; padding: 10px; border-radius: 5px; font-size: 16px;"
        )
        self.start_export_button.clicked.connect(self.handle_export)
        main_layout.addWidget(self.start_export_button)
        main_layout.addSpacing(10)

        # Status area
        self.export_status_text = QTextEdit()
        self.export_status_text.setReadOnly(True)
        self.export_status_text.setMinimumHeight(100)
        self.export_status_text.setStyleSheet("background-color: #fdfefe; border: 1px solid #ccc;")
        main_layout.addWidget(self.export_status_text)

        main_layout.addStretch(1) # Push everything to the top
        page_widget.setLayout(main_layout)

        if not EXPORT_UTILS_AVAILABLE:
            self.start_export_button.setEnabled(False)
            self.export_status_text.setText("Export utilities module failed to load. Export functionality is disabled.")
            for cb in [self.cb_export_jpg, self.cb_export_docx, self.cb_export_pdf_report, self.cb_export_all]:
                cb.setEnabled(False)


    def browse_export_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory", self.export_dir_edit.text())
        if directory:
            self.export_dir_edit.setText(directory)

    def handle_export(self):
        if not EXPORT_UTILS_AVAILABLE:
            self.export_status_text.setText("Export utilities are not available. Cannot export.")
            return

        if not self.current_pil_image or not self.current_analysis_results:
            self.export_status_text.setText("Error: No analysis results available to export. Please run an analysis first.")
            return

        output_dir = self.export_dir_edit.text().strip()
        base_name = self.export_base_name_edit.text().strip()

        if not output_dir or not base_name:
            self.export_status_text.setText("Error: Export directory and base filename cannot be empty.")
            return

        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            self.export_status_text.setText(f"Error creating output directory: {e}")
            return

        self.export_status_text.clear()
        self.export_status_text.append("Starting export process...\n")
        QApplication.processEvents() # Update UI

        export_any_succeeded = False

        if self.cb_export_all.isChecked():
            self.export_status_text.append(f"Attempting to export complete package to {output_dir} with base name {base_name}...")
            try:
                if export_complete_package_util:
                    export_files = export_complete_package_util(self.current_pil_image, self.current_analysis_results, os.path.join(output_dir, base_name))
                    self.export_status_text.append("Comprehensive package export attempted.")
                    for file_type, filename in export_files.items():
                        if filename and os.path.exists(filename):
                            self.export_status_text.append(f"  SUCCESS: {file_type} -> {filename}")
                            export_any_succeeded = True
                        else:
                            self.export_status_text.append(f"  FAILED or SKIPPED: {file_type}")
                else:
                    self.export_status_text.append("  FAILED: export_complete_package function not available.")
            except Exception as e:
                self.export_status_text.append(f"  ERROR during complete package export: {e}")
            self.export_status_text.append("-" * 30 + "\n")
        else:
            # Individual exports
            if self.cb_export_jpg.isChecked():
                png_temp_path = os.path.join(output_dir, f"{base_name}_visualization_temp.png")
                jpg_final_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
                self.export_status_text.append(f"Exporting JPG visualization to {jpg_final_path}...")
                try:
                    if export_visualization_png_util and Image:
                        # First export as PNG (as render_visualization_on_canvas is for GUI figure, not direct file save)
                        png_path = export_visualization_png_util(self.current_pil_image, self.current_analysis_results, png_temp_path)
                        if png_path and os.path.exists(png_path):
                            img_for_jpg = Image.open(png_path)
                            if img_for_jpg.mode == 'RGBA':
                                img_for_jpg = img_for_jpg.convert('RGB')
                            img_for_jpg.save(jpg_final_path, "JPEG", quality=95)
                            os.remove(png_path) # Remove temp PNG
                            self.export_status_text.append(f"  SUCCESS: JPG Visualization -> {jpg_final_path}")
                            export_any_succeeded = True
                        else:
                            self.export_status_text.append("  FAILED: Could not generate intermediate PNG for JPG conversion.")
                    else:
                        self.export_status_text.append("  FAILED: PNG export or PIL/Image module not available for JPG conversion.")
                except Exception as e:
                    self.export_status_text.append(f"  ERROR during JPG export: {e}")
                self.export_status_text.append("-" * 30 + "\n")

            if self.cb_export_docx.isChecked():
                docx_path = os.path.join(output_dir, f"{base_name}_report.docx")
                self.export_status_text.append(f"Exporting DOCX report to {docx_path}...")
                try:
                    if export_to_advanced_docx_util:
                        result_path = export_to_advanced_docx_util(self.current_pil_image, self.current_analysis_results, docx_path)
                        if result_path and os.path.exists(result_path):
                            self.export_status_text.append(f"  SUCCESS: DOCX Report -> {result_path}")
                            export_any_succeeded = True
                        else:
                            self.export_status_text.append("  FAILED: DOCX export function did not return a valid path or file not found.")
                    else:
                        self.export_status_text.append("  FAILED: DOCX export function not available.")
                except Exception as e:
                    self.export_status_text.append(f"  ERROR during DOCX export: {e}")
                self.export_status_text.append("-" * 30 + "\n")

            if self.cb_export_pdf_report.isChecked():
                docx_path_for_pdf = os.path.join(output_dir, f"{base_name}_report.docx")
                pdf_path = os.path.join(output_dir, f"{base_name}_report.pdf")
                self.export_status_text.append(f"Exporting PDF report (from DOCX) to {pdf_path}...")
                try:
                    if export_report_pdf_util:
                        if not os.path.exists(docx_path_for_pdf):
                            self.export_status_text.append(f"  INFO: DOCX file not found at {docx_path_for_pdf}. Attempting to generate it first...")
                            if export_to_advanced_docx_util:
                                export_to_advanced_docx_util(self.current_pil_image, self.current_analysis_results, docx_path_for_pdf)
                            else:
                                raise Exception("DOCX export function not available to create prerequisite for PDF.")

                        if os.path.exists(docx_path_for_pdf):
                            result_path = export_report_pdf_util(docx_path_for_pdf, pdf_path)
                            if result_path and os.path.exists(result_path):
                                self.export_status_text.append(f"  SUCCESS: PDF Report -> {result_path}")
                                export_any_succeeded = True
                            else:
                                self.export_status_text.append("  FAILED: PDF conversion failed. Check if a DOCX to PDF converter (e.g., LibreOffice) is installed and in PATH, or if docx2pdf library is installed.")
                        else:
                            self.export_status_text.append("  FAILED: Could not generate prerequisite DOCX file for PDF conversion.")
                    else:
                         self.export_status_text.append("  FAILED: PDF export function not available.")
                except Exception as e:
                    self.export_status_text.append(f"  ERROR during PDF export: {e}")
                self.export_status_text.append("-" * 30 + "\n")

        if export_any_succeeded:
            self.export_status_text.append("Export process completed. Check files in the specified directory.")
        else:
            self.export_status_text.append("Export process completed, but no files were successfully generated or selected for export.")
        QApplication.processEvents()


    def _parse_step_message(self, full_message):
        # Example: "✅ [1/17] File validation passed" -> "File validation passed"
        # Example: "📊 [5/17] Multi-quality Error Level Analysis..." -> "Multi-quality Error Level Analysis..."
        match = re.search(r"\[\d+/\d+\]\s*(.*)", full_message)
        if match:
            return match.group(1).strip()
        # Fallback for simple messages without the [x/y] prefix
        simple_name_match = re.match(r"[✅🔍🔧📊🎯🔄🧩📡📷🌊🧵📐💡📈🤖🏁]\s*(.*)", full_message)
        if simple_name_match:
            return simple_name_match.group(1).strip()
        return full_message # Return original if no specific pattern matches


    def browse_image(self):
        if not config:
            self.validation_status_label.setText("Configuration module not loaded.")
            self.validation_status_label.setStyleSheet("color: red;")
            self.start_analysis_button.setEnabled(False)
            return

        # Ensure previous analysis isn't running or button is disabled during analysis
        if self.analysis_thread.isRunning():
            self.validation_status_label.setText("Cannot browse: Analysis in progress.")
            self.validation_status_label.setStyleSheet("color: orange;")
            return

        valid_extensions_str = "Images (" + " ".join("*" + ext for ext in config.VALID_EXTENSIONS) + ")"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an Image", "", valid_extensions_str + ";;All Files (*)"
        )

        # Reset button state regardless of selection, will be re-enabled if valid
        self.start_analysis_button.setEnabled(False)

        if file_path:
            self.current_image_path = file_path
            print(f"Selected image: {self.current_image_path}")
            try:
                # Store PIL image for later use in visualization
                if Image:
                    self.current_pil_image = Image.open(self.current_image_path)
                else: # Should not happen if Image is None from the top import try-except
                    self.current_pil_image = None
                    raise ImportError("PIL/Pillow (Image module) is not available to open image.")
            except Exception as e:
                self.current_pil_image = None
                self.validation_status_label.setText(f"Error loading image file: {e}")
                self.validation_status_label.setStyleSheet("color: red;")
                self.start_analysis_button.setEnabled(False)
                return # Stop further processing if image can't be loaded

            self.display_image_preview()
            self.validate_selected_image()
        else:
            self.current_image_path = None
            self.current_pil_image = None
            self.image_preview_label.clear()
            self.image_preview_label.setText("Image Preview Area\n\n(Select an image to see preview)")
            self.validation_status_label.setText("Validation status: N/A")


    def display_image_preview(self):
        if self.current_image_path:
            pixmap = QPixmap(self.current_image_path)
            if pixmap.isNull():
                self.image_preview_label.setText("Cannot preview this image type or file is corrupted.")
                self.validation_status_label.setText("Error: Cannot load image for preview.")
                self.validation_status_label.setStyleSheet("color: red;")
                self.start_analysis_button.setEnabled(False)
                return

            scaled_pixmap = pixmap.scaled(
                self.image_preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_preview_label.setPixmap(scaled_pixmap)
        else:
            self.image_preview_label.clear()
            self.image_preview_label.setText("Image Preview Area\n\n(Select an image to see preview)")
            self.start_analysis_button.setEnabled(False)


    def validate_selected_image(self):
        self.start_analysis_button.setEnabled(False)
        if not self.current_image_path:
            self.validation_status_label.setText("Validation status: No image selected.")
            self.validation_status_label.setStyleSheet("color: orange;")
            return

        if not validation or not config: # Or other critical analysis modules
            self.validation_status_label.setText("Validation/Config modules not loaded.")
            self.validation_status_label.setStyleSheet("color: red;")
            return

        try:
            ext = os.path.splitext(self.current_image_path)[1].lower()
            if ext not in config.VALID_EXTENSIONS:
                raise self.InvalidFileType(f"Unsupported file type: {ext}. Valid: {', '.join(config.VALID_EXTENSIONS)}")

            file_size = os.path.getsize(self.current_image_path)
            if file_size < config.MIN_FILE_SIZE:
                raise self.FileSizeError(f"File size too small ({file_size}B). Min: {config.MIN_FILE_SIZE}B.")

            validation.validate_image_file(self.current_image_path) # From validation.py

            self.validation_status_label.setText("Validation status: Image is valid and ready for analysis.")
            self.validation_status_label.setStyleSheet("color: green;")
            self.start_analysis_button.setEnabled(True) # Enable button if all checks pass

        except (self.InvalidFileType, self.FileSizeError, FileNotFoundError, ValueError) as e:
            self.validation_status_label.setText(f"Validation Error: {e}")
            self.validation_status_label.setStyleSheet("color: red;")
        except Exception as e:
            self.validation_status_label.setText(f"An unexpected validation error: {e}")
            self.validation_status_label.setStyleSheet("color: red;")
            print(f"Unexpected validation error: {e}")

    def start_image_analysis(self):
        if not self.current_image_path:
            self.on_analysis_error("No image selected for analysis.")
            return
        if self.analysis_thread.isRunning():
            self.on_analysis_error("Analysis is already in progress.")
            return

        self.set_ui_analysis_mode(True) # Disable UI elements, show progress bar

        # Clear content from all processing step pages
        for step_idx in self.processing_pages:
            page = self.processing_pages[step_idx]
            title_label = page.findChild(QLabel, "step_title_label")
            img_label = page.findChild(QLabel, "image_display_label")
            text_area = page.findChild(QTextEdit, "text_display_area")
            if title_label: title_label.setText(f"Step {step_idx}: Waiting...")
            if img_label: img_label.clear(); img_label.hide()
            if text_area: text_area.clear(); text_area.hide()

        # Switch to the first processing page (e.g., step 0)
        if 0 in self.processing_pages:
             self.stacked_widget.setCurrentWidget(self.processing_pages[0])
        else:
            self.switch_page("Processing")

        self.output_dir = os.path.join(os.getcwd(), "gui_analysis_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir, exist_ok=True)

        if hasattr(self.analysis_worker, '_is_cancelled'):
            self.analysis_worker._is_cancelled = False
        try: self.analysis_thread.started.disconnect()
        except TypeError: pass # No connection existed or already disconnected

        self.analysis_thread.started.connect(
            lambda: self.analysis_worker.run_analysis(self.current_image_path, self.output_dir)
        )
        self.analysis_thread.start()
        if hasattr(self, 'global_status_label'): self.global_status_label.setText("Analysis in progress...")
        print("Analysis thread started.")

    def request_cancel_analysis(self):
        if self.analysis_thread and self.analysis_thread.isRunning():
            print("GUI: Requesting analysis cancellation...")
            if hasattr(self.analysis_worker, 'cancel'): self.analysis_worker.cancel()
            self.cancel_analysis_button.setEnabled(False)
            if hasattr(self, 'global_status_label'): self.global_status_label.setText("Cancelling analysis...")

    def set_ui_analysis_mode(self, is_analyzing):
        """Helper to enable/disable UI elements during analysis."""
        self.browse_button.setEnabled(not is_analyzing)
        self.start_analysis_button.setEnabled(not is_analyzing)
        self.cancel_analysis_button.setEnabled(is_analyzing)
        self.progress_bar.setVisible(is_analyzing)
        if not is_analyzing:
            self.progress_bar.setValue(0) # Reset progress bar
            self.global_status_label.setText("Ready.")

        for btn_name, nav_btn in self.nav_buttons.items():
            # Allow navigation to "Processing" tab itself, but not others.
            # Or simply disable all nav buttons. For now, disable all.
            nav_btn.setEnabled(not is_analyzing)


    def on_step_started(self, step_number, step_name_full):
        clean_step_name = self._parse_step_message(step_name_full)
        print(f"GUI: Step {step_number} started - {clean_step_name}")
        self.global_status_label.setText(f"Step {step_number}: {clean_step_name}...")
        self.progress_bar.setValue(step_number)


        if step_number in self.processing_pages:
            current_page = self.processing_pages[step_number]
            self.stacked_widget.setCurrentWidget(current_page)

            title_label = current_page.findChild(QLabel, "step_title_label")
            img_label = current_page.findChild(QLabel, "image_display_label")
            text_area = current_page.findChild(QTextEdit, "text_display_area")

            if title_label: title_label.setText(f"Step {step_number}: {clean_step_name} - Processing...")
            if img_label: img_label.clear(); img_label.hide()
            if text_area: text_area.clear(); text_area.hide()
        else:
            print(f"Warning: No page defined for step number {step_number}.")
            # If no specific page, update the generic "Processing" landing page label (if it exists)
            # This might not be the ideal UX if individual pages are expected.
            generic_processing_page = self.pages.get("Processing")
            if generic_processing_page and self.stacked_widget.currentWidget() != generic_processing_page :
                 # Avoid switching if already on a specific processing page.
                 # This part of the logic might need review based on desired flow for out-of-bound steps.
                 pass


    def on_step_completed(self, step_number, message_full, intermediate_result):
        clean_message = self._parse_step_message(message_full)
        print(f"GUI: Step {step_number} completed - {clean_message}")
        self.progress_bar.setValue(step_number + 1) # Show progress after step completion
        self.global_status_label.setText(f"Step {step_number}: {clean_message} - Completed.")


        if step_number not in self.processing_pages:
            print(f"Warning: No page defined for step {step_number} to display results.")
            return

        current_page = self.processing_pages[step_number]
        title_label = current_page.findChild(QLabel, "step_title_label")
        img_label = current_page.findChild(QLabel, "image_display_label")
        text_area = current_page.findChild(QTextEdit, "text_display_area")

        if title_label: title_label.setText(f"Step {step_number}: {clean_message} - Completed")

        img_label.hide(); text_area.hide() # Hide both initially

        if isinstance(intermediate_result, Image.Image if Image else type(None)):
            try:
                # Ensure intermediate_result is RGB before tobytes, common issue for some PIL modes
                if intermediate_result.mode != 'RGB':
                    intermediate_result = intermediate_result.convert('RGB')
                q_image = QImage(intermediate_result.tobytes("raw", "RGB"), intermediate_result.width, intermediate_result.height, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                img_label.setPixmap(pixmap.scaled(img_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                img_label.show()
            except Exception as e:
                print(f"Error converting PIL Image to QPixmap for step {step_number}: {e}")
                text_area.setText(f"Could not display PIL image.\nError: {e}\nData: {str(intermediate_result)[:200]}...")
                text_area.show()
        elif isinstance(intermediate_result, np.ndarray if np else type(None)):
            try:
                arr_uint8 = np.require(intermediate_result, dtype=np.uint8, requirements='C')
                if arr_uint8.ndim == 2:
                    h, w = arr_uint8.shape
                    q_img = QImage(arr_uint8.data, w, h, w, QImage.Format.Format_Grayscale8)
                elif arr_uint8.ndim == 3 and arr_uint8.shape[2] in [3, 4]:
                    h, w, ch = arr_uint8.shape
                    fmt = QImage.Format.Format_RGB888 if ch == 3 else QImage.Format.Format_RGBA8888
                    q_img = QImage(arr_uint8.data, w, h, w * ch, fmt)
                else:
                    raise ValueError(f"Unsupported numpy array shape: {arr_uint8.shape}")

                pixmap = QPixmap.fromImage(q_img)
                img_label.setPixmap(pixmap.scaled(img_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                img_label.show()
            except Exception as e:
                print(f"Error converting NumPy array to QPixmap for step {step_number}: {e}")
                text_area.setText(f"NumPy array (shape: {intermediate_result.shape}, dtype: {intermediate_result.dtype}). Preview failed: {e}")
                text_area.show()
        elif isinstance(intermediate_result, (dict, list)):
            try:
                text_area.setText(pprint.pformat(intermediate_result, indent=2, width=100))
                text_area.show()
            except Exception as e:
                text_area.setText(f"Error formatting dict/list: {e}\nData: {str(intermediate_result)[:200]}...")
                text_area.show()
        elif isinstance(intermediate_result, (str, int, float, bool)):
            text_area.setText(str(intermediate_result))
            text_area.show()
        elif intermediate_result is None:
            text_area.setText("No specific result data for this step.")
            text_area.show()
        else:
            text_area.setText(f"Unknown data type: {type(intermediate_result)}\nData: {str(intermediate_result)[:200]}...")
            text_area.show()


    def on_analysis_finished(self, final_results):
        print("GUI: Analysis finished!")
        self.current_analysis_results = final_results
        self.set_ui_analysis_mode(False) # Re-enable UI
        self.progress_bar.setValue(self.progress_bar.maximum()) # Fill progress bar
        self.global_status_label.setText("Analysis completed successfully.")

        if database and self.current_image_path and self.output_dir:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            img_name = os.path.basename(self.current_image_path)
            classification_res = final_results.get('classification', {})
            serializable_results = {k: v for k, v in final_results.items()
                                    if not isinstance(v, (Image.Image if Image else type(None), np.ndarray if np else type(None)))}
            database.save_analysis(
                timestamp=ts, image_name=img_name, image_path=self.current_image_path,
                classification_type=classification_res.get('type', 'N/A'),
                confidence_score=classification_res.get('confidence', 0.0),
                copy_move_score=classification_res.get('copy_move_score', 0.0),
                splicing_score=classification_res.get('splicing_score', 0.0),
                output_dir=self.output_dir, full_results_dict=serializable_results
            )
            if self.stacked_widget.currentWidget() == self.pages.get("History"):
                self.load_analysis_history()

        self.analysis_thread.quit()
        self.analysis_thread.wait()

        self.switch_page("Results")

        if self.results_summary_text:
            if self.current_analysis_results and 'classification' in self.current_analysis_results:
                # ... (summary text population as before) ...
                classification_info = self.current_analysis_results['classification'] # Duplicated but ok for clarity
                summary_text = (f"<b>ANALYSIS COMPLETE for: {os.path.basename(self.current_image_path)}</b><br><br>"
                                # ... (rest of the summary text formatting) ...
                                f"<b>Output Directory:</b> {self.output_dir}<br><br>"
                                f"<b>Details:</b><br>{'<br>'.join(classification_info.get('details', []))}")
                self.results_summary_text.setHtml(summary_text)


                if self.matplotlib_available and self.current_pil_image and self.results_figure:
                    try:
                        print("Rendering Matplotlib visualization on canvas...")
                        render_visualization_on_canvas(self.results_figure, self.current_pil_image, self.current_analysis_results)
                        self.results_canvas.draw()
                        print("Matplotlib canvas drawn.")
                    except Exception as e_vis:
                        print(f"Error rendering visualization: {e_vis}")
                        if isinstance(self.results_canvas, QLabel):
                           self.results_canvas.setText(f"Error rendering Matplotlib visualization: {e_vis}")
            else:
                self.results_summary_text.setText("Analysis finished, but results structure is unexpected or missing classification.")

        print("Analysis thread finished and joined.")

    def on_analysis_error(self, error_message):
        print(f"GUI: Analysis error - {error_message}")
        self.set_ui_analysis_mode(False) # Re-enable UI
        self.progress_bar.setValue(0) # Reset progress bar on error

        if "Analysis Cancelled by User" in error_message:
            self.global_status_label.setText("Analysis Cancelled.")
            self.validation_status_label.setText("Analysis Cancelled by User.")
            self.validation_status_label.setStyleSheet("color: orange;")
        else:
            self.global_status_label.setText("Analysis failed.")
            self.validation_status_label.setText(f"Analysis failed: {error_message.splitlines()[0]}")
            self.validation_status_label.setStyleSheet("color: red;")

        if self.analysis_thread.isRunning(): # Check if it's running before trying to quit
            self.analysis_thread.quit()
            self.analysis_thread.wait()

        self.switch_page("Upload")
        print("Analysis thread error handled, quit and waited if running.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'image_preview_label') and self.image_preview_label.isVisible() and self.current_image_path:
            self.display_image_preview()

        # Also rescale image on current processing page if visible
        current_widget = self.stacked_widget.currentWidget()
        if current_widget in self.processing_pages.values():
            img_label = current_widget.findChild(QLabel, "image_display_label")
            if img_label and img_label.pixmap() and not img_label.pixmap().isNull():
                 # This assumes the intermediate_result that generated the pixmap is still relevant
                 # For simplicity, we're just rescaling the existing pixmap.
                 # A more robust way would be to re-fetch/re-convert the data if needed.
                 original_pixmap = img_label.property("original_pixmap") # We'd need to store this
                 if original_pixmap:
                    scaled_pixmap = original_pixmap.scaled(img_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    img_label.setPixmap(scaled_pixmap)


    def switch_page(self, page_name_or_widget):
        if isinstance(page_name_or_widget, QWidget):
            target_widget = page_name_or_widget
            page_name = "" # For debug print
            for name, widget_obj in self.pages.items(): # Check main pages
                if widget_obj == target_widget:
                    page_name = name
                    break
            if not page_name: # Check processing pages
                 for num, widget_obj in self.processing_pages.items():
                      if widget_obj == target_widget:
                           page_name = f"Processing Step {num}"
                           break
        elif page_name_or_widget in self.pages:
            target_widget = self.pages[page_name_or_widget]
            page_name = page_name_or_widget
            if page_name == "History" and database:
                self.load_analysis_history()
            elif page_name == "Export":
                if self.current_analysis_results and self.current_image_path:
                    self.start_export_button.setEnabled(EXPORT_UTILS_AVAILABLE)
                    self.export_status_text.setText("Ready to export. Select options and click 'Start Export'.")
                    base_name_suggestion = os.path.splitext(os.path.basename(self.current_image_path))[0]
                    self.export_base_name_edit.setText(f"{base_name_suggestion}_analysis")
                else:
                    self.start_export_button.setEnabled(False)
                    self.export_status_text.setText("No analysis results available to export. Please run an analysis first.")
                    self.export_base_name_edit.setText("analysis_export")

        else:
            print(f"Page key '{page_name_or_widget}' not found in main pages.")
            target_widget = self.pages.get("Upload")
            page_name = "Upload (fallback)" # Should not happen if nav buttons are correct

        if target_widget:
            self.stacked_widget.setCurrentWidget(target_widget)
            if page_name == "Upload" and self.current_image_path and hasattr(self, 'image_preview_label'):
                self.display_image_preview() # Refresh preview if going back to upload
            print(f"Switched to {page_name if page_name else 'a processing page'}.")
        else:
            print(f"Could not find widget for page '{page_name_or_widget}'.")

    def load_analysis_history(self):
        if not database:
            print("Database module not available, cannot load history.")
            # Optionally show message in table or a status bar
            if hasattr(self, 'history_table'):
                self.history_table.setRowCount(1)
                self.history_table.setColumnCount(1)
                item = QTableWidgetItem("Database module not available.")
                self.history_table.setItem(0,0, item)
                self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            return
        if not hasattr(self, 'history_table'):
            print("History table not initialized yet.")
            return

        print("Loading analysis history...")
        try:
            records = database.get_all_analyses()
            self.history_table.setRowCount(0) # Clear existing rows
            self.history_table.setColumnCount(7) # Ensure correct column count
            self.history_table.setHorizontalHeaderLabels([
                "Timestamp", "Image Name", "Classification",
                "Confidence", "Copy-Move Score", "Splicing Score", "Output Directory"
            ])


            for row_idx, record in enumerate(records):
                self.history_table.insertRow(row_idx)
                self.history_table.setItem(row_idx, 0, QTableWidgetItem(str(record.get('timestamp','N/A'))))
                self.history_table.setItem(row_idx, 1, QTableWidgetItem(str(record.get('image_name','N/A'))))
                self.history_table.setItem(row_idx, 2, QTableWidgetItem(str(record.get('classification_type','N/A'))))
                self.history_table.setItem(row_idx, 3, QTableWidgetItem(f"{record.get('confidence_score',0.0):.2f}"))
                self.history_table.setItem(row_idx, 4, QTableWidgetItem(f"{record.get('copy_move_score',0.0):.1f}"))
                self.history_table.setItem(row_idx, 5, QTableWidgetItem(f"{record.get('splicing_score',0.0):.1f}"))
                self.history_table.setItem(row_idx, 6, QTableWidgetItem(str(record.get('output_dir','N/A'))))

            # Adjust row heights to content after populating
            self.history_table.resizeRowsToContents()
            print(f"Loaded {len(records)} history records into table.")

        except Exception as e:
            print(f"Error loading history into table: {e}")
            # Display error in table
            self.history_table.setRowCount(1)
            self.history_table.setColumnCount(1)
            item = QTableWidgetItem(f"Error loading history: {e}")
            self.history_table.setItem(0,0, item)


    def closeEvent(self, event):
        if self.analysis_thread.isRunning():
            self.analysis_worker.stop_analysis()
            self.analysis_thread.quit()
            self.analysis_thread.wait(5000)
        event.accept()


# ==============================================================================
# VISUALIZATION FUNCTIONS (Adapted from visualization.py)
# ==============================================================================

# Helper functions (copied from visualization.py, may need minor adaptation if they use global plt)
# For brevity, I will only copy the main visualize_results_advanced and assume helpers are either
# self-contained enough or will be adapted similarly. In a real scenario, all helpers would be here.

def render_visualization_on_canvas(figure, original_pil, analysis_results):
    """Renders the comprehensive visualization onto a provided Matplotlib FigureCanvas."""
    if not MATPLOTLIB_AVAILABLE:
        print("Attempted to render visualization, but Matplotlib is not available.")
        return
    if not original_pil:
        print("Original PIL image not available for visualization.")
        return
    if not analysis_results:
        print("Analysis results not available for visualization.")
        return

    figure.clear() # Clear the figure before drawing

    try:
        # Most of the content of visualize_results_advanced from visualization.py goes here
        # with 'fig' replaced by 'figure' and plt. calls adapted.
        # gs = figure.add_gridspec(4, 5, hspace=0.3, wspace=0.2) # Example, adjust as needed
        gs = gridspec.GridSpecFromSubplotSpec(4, 5, figure.add_subplot(111), hspace=0.3, wspace=0.2)
        figure.clf() # Clear figure again, add_gridspec might behave weirdly with existing axes from add_subplot(111)
        gs = figure.add_gridspec(4, 5, hspace=0.3, wspace=0.2)


        classification = analysis_results.get('classification', {})

        figure.suptitle(
            f"Advanced Forensic Image Analysis Report\n"
            f"Analysis Complete - Classification: {classification.get('type', 'N/A')}",
            fontsize=16, fontweight='bold'
        )

        # Row 1: Basic Analysis
        ax1 = figure.add_subplot(gs[0, 0])
        ax1.imshow(original_pil)
        ax1.set_title("Original Image", fontsize=11)
        ax1.axis('off')

        ax2 = figure.add_subplot(gs[0, 1])
        if analysis_results.get('ela_image') is not None:
            ela_display = ax2.imshow(analysis_results['ela_image'], cmap='hot')
            ax2.set_title(f"Multi-Quality ELA\n(μ={analysis_results.get('ela_mean',0):.1f}, σ={analysis_results.get('ela_std',0):.1f})", fontsize=11)
            figure.colorbar(ela_display, ax=ax2, fraction=0.046, pad=0.04) # Added pad
        else:
            ax2.text(0.5, 0.5, "ELA Data\nNot Available", ha='center', va='center')
        ax2.axis('off')

        ax3 = figure.add_subplot(gs[0, 2])
        # Assuming create_feature_match_visualization_gui is adapted and available
        create_feature_match_visualization_gui(ax3, original_pil, analysis_results)

        ax4 = figure.add_subplot(gs[0, 3])
        create_block_match_visualization_gui(ax4, original_pil, analysis_results)

        ax5 = figure.add_subplot(gs[0, 4])
        create_kmeans_clustering_visualization_gui(ax5, original_pil, analysis_results, figure) # Pass figure for sub-gridspec

        # Row 2: Advanced Analysis
        ax6 = figure.add_subplot(gs[1, 0])
        create_frequency_visualization_gui(ax6, analysis_results)

        ax7 = figure.add_subplot(gs[1, 1])
        create_texture_visualization_gui(ax7, analysis_results)

        ax8 = figure.add_subplot(gs[1, 2])
        create_edge_visualization_gui(ax8, original_pil, analysis_results)

        ax9 = figure.add_subplot(gs[1, 3])
        create_illumination_visualization_gui(ax9, original_pil, analysis_results)

        ax10 = figure.add_subplot(gs[1, 4])
        if analysis_results.get('jpeg_ghost') is not None:
            ghost_display = ax10.imshow(analysis_results['jpeg_ghost'], cmap='hot')
            ax10.set_title(f"JPEG Ghost\n({analysis_results.get('jpeg_ghost_suspicious_ratio',0):.1%} suspicious)", fontsize=11)
            figure.colorbar(ghost_display, ax=ax10, fraction=0.046, pad=0.04) # Added pad
        else:
            ax10.text(0.5,0.5, "JPEG Ghost\nNot Available", ha='center',va='center')
        ax10.axis('off')

        # Row 3: Statistical Analysis
        ax11 = figure.add_subplot(gs[2, 0])
        create_statistical_visualization_gui(ax11, analysis_results)

        ax12 = figure.add_subplot(gs[2, 1])
        if analysis_results.get('noise_map') is not None:
            ax12.imshow(analysis_results['noise_map'], cmap='gray')
            ax12.set_title(f"Noise Map\n(Inconsistency: {analysis_results.get('noise_analysis',{}).get('overall_inconsistency',0):.3f})", fontsize=11)
        else:
            ax12.text(0.5,0.5, "Noise Map\nNot Available", ha='center',va='center')
        ax12.axis('off')

        ax13 = figure.add_subplot(gs[2, 2])
        create_quality_response_plot_gui(ax13, analysis_results)

        ax14 = figure.add_subplot(gs[2, 3])
        combined_heatmap = create_advanced_combined_heatmap_gui(analysis_results, original_pil.size)
        if combined_heatmap is not None:
            ax14.imshow(combined_heatmap, cmap='hot', alpha=0.7)
            ax14.imshow(original_pil, alpha=0.3)
        else:
            ax14.imshow(original_pil) # Show original if heatmap failed
            ax14.text(0.5,0.5, "Heatmap Error", color='red', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        ax14.set_title("Combined Suspicion Heatmap", fontsize=11)
        ax14.axis('off')

        ax15 = figure.add_subplot(gs[2, 4])
        create_technical_metrics_plot_gui(ax15, analysis_results)

        # Row 4: Detailed Analysis Report Text (simplified for GUI)
        ax16 = figure.add_subplot(gs[3, :])
        create_detailed_report_gui(ax16, analysis_results)

    except Exception as e:
        figure.clf() # Clear figure on error
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, f"Error generating visualization:\n{str(e)}",
                ha='center', va='center', color='red', fontsize=12, wrap=True)
        print(f"Error in render_visualization_on_canvas: {e}")
        import traceback
        traceback.print_exc()

# Placeholder for adapted helper functions (these would also be copied and modified from visualization.py)
# Each of these would take 'ax' as an argument and use it for plotting instead of plt.
def create_feature_match_visualization_gui(ax, original_pil, results):
    # Adapted from visualization.py
    # Ensure np, cv2 are available (checked by top-level MATPLOTLIB_AVAILABLE effectively)
    if not (np and cv2 and Image): ax.text(0.5,0.5,"Feature Viz Disabled\n(Missing np/cv2/PIL)", ha='center'); ax.axis('off'); return
    img_matches = np.array(original_pil.convert('RGB'))
    sift_keypoints = results.get('sift_keypoints')
    ransac_matches = results.get('ransac_matches')

    if sift_keypoints and ransac_matches:
        keypoints = sift_keypoints
        matches = ransac_matches[:20]
        for match in matches:
            if hasattr(match, 'queryIdx') and hasattr(match, 'trainIdx'):
                 pt1 = tuple(map(int, keypoints[match.queryIdx].pt))
                 pt2 = tuple(map(int, keypoints[match.trainIdx].pt))
                 cv2.line(img_matches, pt1, pt2, (0, 255, 0), 2)
                 cv2.circle(img_matches, pt1, 5, (255, 0, 0), -1)
                 cv2.circle(img_matches, pt2, 5, (255, 0, 0), -1)
    ax.imshow(img_matches)
    ax.set_title(f"RANSAC Matches ({results.get('ransac_inliers',0)} inliers)", fontsize=9)
    ax.axis('off')

def create_block_match_visualization_gui(ax, original_pil, results):
    if not (np and cv2 and Image): ax.text(0.5,0.5,"Block Viz Disabled", ha='center'); ax.axis('off'); return
    img_blocks = np.array(original_pil.convert('RGB'))
    block_matches = results.get('block_matches', [])
    if block_matches: # Check if block_matches is not None
        for i, match in enumerate(block_matches[:15]):
            x1, y1 = match['block1']
            x2, y2 = match['block2']
            color = (255,0,0) if i%2==0 else (0,255,0)
            cv2.rectangle(img_blocks, (x1,y1), (x1+16,y1+16), color, 2)
            cv2.rectangle(img_blocks, (x2,y2), (x2+16,y2+16), color, 2)
            cv2.line(img_blocks, (x1+8,y1+8), (x2+8,y2+8), (255,255,0),1)
    ax.imshow(img_blocks)
    ax.set_title(f"Block Matches ({len(block_matches if block_matches else [])} found)", fontsize=9)
    ax.axis('off')

def create_kmeans_clustering_visualization_gui(ax, original_pil, analysis_results, main_figure):
    if not (np and cv2 and Image and MATPLOTLIB_AVAILABLE and gridspec): ax.text(0.5,0.5,"KMeans Viz Disabled", ha='center'); ax.axis('off'); return
    loc_results = analysis_results.get('localization_analysis', {})
    kmeans_data = loc_results.get('kmeans_localization', {})
    if not kmeans_data or 'localization_map' not in kmeans_data:
        ax.imshow(original_pil); ax.set_title("K-means Data Missing", fontsize=9); ax.axis('off'); return

    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec(), hspace=0.2, wspace=0.1)
    ax.clear(); ax.axis('off') # Clear original ax

    ax1 = main_figure.add_subplot(gs_sub[0, 0])
    cluster_map = kmeans_data['localization_map']
    n_clusters = len(np.unique(cluster_map)) if cluster_map is not None else 0
    if cluster_map is not None: cluster_display = ax1.imshow(cluster_map, cmap='tab10', alpha=0.8)
    ax1.imshow(original_pil, alpha=0.2)
    ax1.set_title(f"K-means Clusters (n={n_clusters})", fontsize=8); ax1.axis('off')
    if cluster_map is not None: main_figure.colorbar(cluster_display, ax=ax1, fraction=0.046, pad=0.04).set_label('Cluster ID', fontsize=7)

    ax2 = main_figure.add_subplot(gs_sub[0, 1])
    tampering_highlight = np.zeros_like(cluster_map) if cluster_map is not None else None
    tampering_cluster_id = kmeans_data.get('tampering_cluster_id', -1)
    if tampering_highlight is not None and tampering_cluster_id != -1: tampering_highlight[cluster_map == tampering_cluster_id] = 1
    ax2.imshow(original_pil)
    if tampering_highlight is not None: ax2.imshow(tampering_highlight, cmap='Reds', alpha=0.6)
    ax2.set_title(f"Tampering Cluster (ID={tampering_cluster_id})", fontsize=8); ax2.axis('off')

    ax.set_title("K-means Localization", fontsize=10) # Title for the group of subplots

# ... (Other helper plot functions adapted similarly: create_frequency_visualization_gui, etc.) ...
# For brevity, I'm omitting the full adaptation of every single helper here,
# but they would follow the pattern of taking `ax` and using it.
def create_frequency_visualization_gui(ax, results):
    if not MATPLOTLIB_AVAILABLE: return
    freq_data = results.get('frequency_analysis', {}).get('dct_stats', {})
    categories = ['Low Freq', 'Mid Freq', 'High Freq']
    values = [freq_data.get('low_freq_energy',0), freq_data.get('mid_freq_energy',0), freq_data.get('high_freq_energy',0)]
    ax.bar(categories, values, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_title(f"Frequency Domain (Incon: {results.get('frequency_analysis',{}).get('frequency_inconsistency',0):.2f})", fontsize=9)
    ax.set_ylabel('Energy', fontsize=8); ax.tick_params(axis='both', which='major', labelsize=7)

def create_texture_visualization_gui(ax, results):
    if not MATPLOTLIB_AVAILABLE: return
    texture_data = results.get('texture_analysis',{}).get('texture_consistency',{})
    metrics = list(texture_data.keys())
    values = list(texture_data.values())
    ax.barh(metrics, values, color='purple', alpha=0.7)
    ax.set_title(f"Texture Consistency (Overall: {results.get('texture_analysis',{}).get('overall_inconsistency',0):.3f})", fontsize=9)
    ax.set_xlabel('Inconsistency Score', fontsize=8); ax.tick_params(axis='both', which='major', labelsize=7)

def create_edge_visualization_gui(ax, original_pil, results):
    if not (Image and np and sobel and MATPLOTLIB_AVAILABLE): ax.text(0.5,0.5,"Edge Viz Disabled", ha='center'); ax.axis('off'); return
    image_gray = np.array(original_pil.convert('L'))
    edges = sobel(image_gray)
    ax.imshow(edges, cmap='gray')
    ax.set_title(f"Edge Analysis (Incon: {results.get('edge_analysis',{}).get('edge_inconsistency',0):.3f})", fontsize=9)
    ax.axis('off')

def create_illumination_visualization_gui(ax, original_pil, results):
    if not (Image and np and cv2 and MATPLOTLIB_AVAILABLE): ax.text(0.5,0.5,"Illum Viz Disabled", ha='center'); ax.axis('off'); return
    image_array = np.array(original_pil)
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    illumination = lab[:, :, 0]
    ax.imshow(illumination, cmap='gray')
    ax.set_title(f"Illumination Map (Incon: {results.get('illumination_analysis',{}).get('overall_illumination_inconsistency',0):.3f})", fontsize=9)
    ax.axis('off')

def create_statistical_visualization_gui(ax, results):
    if not MATPLOTLIB_AVAILABLE: return
    stats = results.get('statistical_analysis',{})
    channels = ['R', 'G', 'B']
    entropies = [stats.get(f'{ch}_entropy',0) for ch in channels]
    ax.bar(channels, entropies, color=['red', 'green', 'blue'], alpha=0.7)
    ax.set_title(f"Channel Entropies (Overall: {stats.get('overall_entropy',0):.3f})", fontsize=9)
    ax.set_ylabel('Entropy', fontsize=8); ax.tick_params(axis='both', which='major', labelsize=7)

def create_quality_response_plot_gui(ax, results):
    if not MATPLOTLIB_AVAILABLE: return
    jpeg_analysis_data = results.get('jpeg_analysis', {})
    quality_responses = jpeg_analysis_data.get('quality_responses', [])
    qualities = [r['quality'] for r in quality_responses]
    responses = [r['response_mean'] for r in quality_responses]
    if qualities and responses:
        ax.plot(qualities, responses, 'b-o', linewidth=1, markersize=3)
    ax.set_title(f"JPEG Quality Response (Est. Orig: {jpeg_analysis_data.get('estimated_original_quality', 'N/A')})", fontsize=9)
    ax.set_xlabel('Quality', fontsize=8); ax.set_ylabel('Response', fontsize=8)
    ax.grid(True, alpha=0.3); ax.tick_params(axis='both', which='major', labelsize=7)

def create_advanced_combined_heatmap_gui(analysis_results, image_size):
    if not (np and cv2 and Image): return None # Cannot create heatmap
    w, h = image_size
    heatmap = np.zeros((h, w), dtype=np.float32)

    ela_image = analysis_results.get('ela_image')
    if ela_image is not None and hasattr(ela_image, 'resize'):
        ela_resized = np.array(ela_image.resize((w,h), Image.Resampling.LANCZOS if hasattr(Image,'Resampling') else Image.LANCZOS)) / 255.0
        heatmap += ela_resized * 0.3

    jpeg_ghost = analysis_results.get('jpeg_ghost')
    if jpeg_ghost is not None:
        ghost_resized = cv2.resize(jpeg_ghost, (w, h))
        heatmap += ghost_resized * 0.25 # Assuming ghost_resized is already 0-1

    return np.clip(heatmap, 0, 1)


def create_technical_metrics_plot_gui(ax, results):
    if not MATPLOTLIB_AVAILABLE: return
    metrics = ['ELA Mean', 'RANSAC', 'Blocks', 'Noise', 'JPEG Ghost']
    values = [
        results.get('ela_mean',0),
        results.get('ransac_inliers',0),
        len(results.get('block_matches',[]) if results.get('block_matches') is not None else []),
        results.get('noise_analysis',{}).get('overall_inconsistency',0) * 100,
        results.get('jpeg_ghost_suspicious_ratio',0) * 100
    ]
    ax.bar(metrics, values, color=['orange', 'green', 'blue', 'red', 'purple'], alpha=0.8)
    ax.set_title("Tech Metrics Summary", fontsize=9)
    ax.set_ylabel('Score/Count', fontsize=8); ax.tick_params(axis='x', rotation=30, labelsize=7); ax.tick_params(axis='y', labelsize=7)

def create_detailed_report_gui(ax, analysis_results):
    if not MATPLOTLIB_AVAILABLE: return
    ax.axis('off')
    classification = analysis_results.get('classification', {})
    metadata = analysis_results.get('metadata', {})
    report_text = f"CLASSIFICATION: {classification.get('type', 'N/A')} (Conf: {classification.get('confidence', 'N/A')})\n"
    report_text += f"Copy-Move: {classification.get('copy_move_score','N/A')}/100, Splicing: {classification.get('splicing_score','N/A')}/100\n"
    report_text += f"DETAILS: {', '.join(classification.get('details',[]))}\n\n"
    report_text += f"METADATA AUTHENTICITY: {metadata.get('Metadata_Authenticity_Score','N/A')}/100\n"
    report_text += f"  Inconsistencies: {len(metadata.get('Metadata_Inconsistency',[]))}\n"
    report_text += f"ELA: μ={analysis_results.get('ela_mean',0):.1f}, σ={analysis_results.get('ela_std',0):.1f}\n"
    # ... add more key details concisely
    ax.text(0.01, 0.99, report_text, transform=ax.transAxes, fontsize=7, va='top', fontfamily='monospace')


# ==============================================================================

def analyze_image_for_gui(image_path, output_dir, worker_emitter: AnalysisWorker):
    """
    Modified version of analyze_image_comprehensive_advanced for GUI integration.
    Emits signals via worker_emitter.
    """
    if not worker_emitter or not worker_emitter.is_running: return None

    step_num = 0
    analysis_results = {} # Initialize results dict

    try:
        worker_emitter.step_started_signal.emit(step_num, "Starting Analysis")
        start_time = time.time()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        worker_emitter.step_completed_signal.emit(step_num, "Setup complete", None)
        step_num += 1

        # 1. Validation
        if not worker_emitter.is_running: return None
        worker_emitter.step_started_signal.emit(step_num, "[1/17] File Validation")
        validation.validate_image_file(image_path) # Uses validation.py
        worker_emitter.step_completed_signal.emit(step_num, "[1/17] Validation passed", None)
        step_num += 1

        # 2. Load image
        if not worker_emitter.is_running: return None
        worker_emitter.step_started_signal.emit(step_num, "[2/17] Image Loading")
        original_image = Image.open(image_path) if Image else None
        if original_image is None: raise ImportError("PIL/Pillow (Image module) not available for loading image.")
        worker_emitter.step_completed_signal.emit(step_num, f"[2/17] Loaded: {os.path.basename(image_path)} ({original_image.size})", None)
        analysis_results['original_image_size'] = original_image.size
        analysis_results['original_image_mode'] = original_image.mode
        step_num += 1

        # 3. Enhanced metadata extraction
        if not worker_emitter.is_running: return None
        worker_emitter.step_started_signal.emit(step_num, "[3/17] Metadata Extraction")
        metadata = validation.extract_enhanced_metadata(image_path)
        analysis_results['metadata'] = metadata
        worker_emitter.step_completed_signal.emit(step_num, f"[3/17] Metadata Authenticity: {metadata.get('Metadata_Authenticity_Score', 'N/A')}/100", metadata)
        step_num += 1

        # 4. Advanced preprocessing
        if not worker_emitter.is_running: return None
        worker_emitter.step_started_signal.emit(step_num, "[4/17] Image Preprocessing")
        if not validation or not hasattr(validation, 'advanced_preprocess_image'):
            raise ImportError("validation.advanced_preprocess_image is not available.")
        preprocessed, original_preprocessed = validation.advanced_preprocess_image(original_image.copy())
        analysis_results['preprocessed_image_shape'] = preprocessed.size
        worker_emitter.step_completed_signal.emit(step_num, "[4/17] Preprocessing done", preprocessed.copy() if preprocessed else None)
        step_num += 1

        # Analysis functions (ELA, Features, etc.)
        analysis_step_defs = [
            (1, "Multi-quality ELA", perform_multi_quality_ela, [preprocessed.copy() if preprocessed else None],
                lambda r: {'ela_image': r[0], 'ela_mean': r[1], 'ela_std': r[2], 'ela_regional': r[3], 'ela_quality_stats': r[4], 'ela_variance': r[5]},
                lambda r: f"ELA Stats: μ={r[1]:.2f}, σ={r[2]:.2f}"),
            (2, "Multi-detector Features", extract_multi_detector_features, [preprocessed.copy()if preprocessed else None, analysis_results.get('ela_image'), analysis_results.get('ela_mean'), analysis_results.get('ela_std')],
                lambda r: {'feature_sets': r[0], 'roi_mask': r[1], 'gray_enhanced': r[2]},
                lambda r: f"Total keypoints: {sum(len(kp) for kp, _ in r[0].values()) if r[0] else 0}"),
            (3, "Advanced Copy-Move Detection", detect_copy_move_advanced, [analysis_results.get('feature_sets'), preprocessed.size if preprocessed else (0,0)],
                lambda r: {'ransac_matches': r[0], 'ransac_inliers': r[1], 'transform': r[2]},
                lambda r: f"RANSAC inliers: {r[1]}"),
            (4, "Enhanced Block Matching", detect_copy_move_blocks, [preprocessed.copy() if preprocessed else None],
                lambda r: {'block_matches': r},
                lambda r: f"Block matches: {len(r) if r else 0}"),
            (5, "Noise Consistency Analysis", analyze_noise_consistency, [preprocessed.copy() if preprocessed else None],
                lambda r: {'noise_analysis': r},
                lambda r: f"Noise inconsistency: {r['overall_inconsistency']:.3f}" if r else "N/A"),
            (6, "Advanced JPEG Analysis", advanced_jpeg_analysis, [preprocessed.copy() if preprocessed else None],
                lambda r: {'jpeg_analysis': r},
                lambda r: f"JPEG Est. Quality: {r.get('estimated_original_quality', 'N/A')}" if r else "N/A"),
            (7, "JPEG Ghost Analysis", jpeg_ghost_analysis, [preprocessed.copy() if preprocessed else None],
                 lambda r: {'jpeg_ghost_map': r[0], 'jpeg_ghost_suspicious': r[1], 'jpeg_ghost_details': r[2] if len(r) > 2 else {}},
                 lambda r: f"Ghost suspicious: {np.sum(r[1]) / r[1].size * 100 if r and r[1] is not None and r[1].size > 0 and np else 0:.1f}%"),
            (8, "Frequency Domain Analysis", analyze_frequency_domain, [preprocessed.copy() if preprocessed else None],
                lambda r: {'frequency_analysis': r},
                lambda r: f"Freq. inconsistency: {r['frequency_inconsistency']:.3f}" if r else "N/A"),
            (9, "Texture Consistency Analysis", analyze_texture_consistency, [preprocessed.copy() if preprocessed else None],
                lambda r: {'texture_analysis': r},
                lambda r: f"Texture inconsistency: {r['overall_inconsistency']:.3f}" if r else "N/A"),
            (10, "Edge Consistency Analysis", analyze_edge_consistency, [preprocessed.copy() if preprocessed else None],
                lambda r: {'edge_analysis': r},
                lambda r: f"Edge inconsistency: {r['edge_inconsistency']:.3f}" if r else "N/A"),
            (11, "Illumination Consistency", analyze_illumination_consistency, [preprocessed.copy() if preprocessed else None],
                lambda r: {'illumination_analysis': r},
                lambda r: f"Illum. inconsistency: {r['overall_illumination_inconsistency']:.3f}" if r else "N/A"),
            (12, "Statistical Analysis", perform_statistical_analysis, [preprocessed.copy() if preprocessed else None],
                lambda r: {'statistical_analysis': r},
                lambda r: f"Overall entropy: {r['overall_entropy']:.3f}" if r else "N/A"),
        ]

        # Adjust step_num to align with the 1-17 main analysis steps for messages
        # Current step_num is 4 (after preproc). The first analysis function is "step 5" in the original script.
        # The step_idx here is 1-based for display in messages like "[5/17]"

        base_step_num_for_analysis_loop = step_num # Save current step_num before loop

        for idx, (orig_step_num_disp, name, func, args_list, result_mapper, summary_func) in enumerate(analysis_step_defs):
            current_processing_step_num = base_step_num_for_analysis_loop + idx
            step_msg_prefix = f"[{orig_step_num_disp+4}/17]" # Original main.py was 1-based, our loop is 0-based, +4 initial steps

            if not worker_emitter.is_running: return None

            # Check if the analysis function itself is available (due to import errors)
            if func is None:
                worker_emitter.step_started_signal.emit(current_processing_step_num, f"{step_msg_prefix} {name}")
                worker_emitter.step_completed_signal.emit(current_processing_step_num, f"{step_msg_prefix} Skipped: {name} module not loaded.", None)
                # Ensure keys are added to analysis_results with None values
                # This assumes result_mapper({}) would give keys if result was an empty dict.
                # A bit risky, might need a safer way to get expected keys.
                try: analysis_results.update({k: None for k in result_mapper({}).keys()})
                except: pass # If result_mapper fails with empty input
                continue

            worker_emitter.step_started_signal.emit(current_processing_step_num, f"{step_msg_prefix} {name}")

            # Argument preparation: handle cases where previous steps might have failed and produced None
            current_args = []
            arg_valid = True
            for arg_val in args_list:
                if isinstance(arg_val, Image.Image if Image else type(None)) or isinstance(arg_val, np.ndarray if np else type(None)):
                    current_args.append(arg_val.copy() if arg_val is not None else None)
                else:
                    current_args.append(arg_val)
                if arg_val is None and args_list.index(arg_val) == 0 : # If primary input (like preprocessed image) is None
                     # Check specific critical args, e.g. first arg is often the image
                    if name not in ["Advanced Copy-Move Detection", "Multi-detector Features"]: # These can somewhat handle None features_sets/ELA
                        arg_valid = False
                        break

            if not arg_valid:
                worker_emitter.step_completed_signal.emit(current_processing_step_num, f"{step_msg_prefix} Skipped: Input data missing for {name}.", None)
                try: analysis_results.update({k: None for k in result_mapper({}).keys()})
                except: pass
                continue

            try:
                raw_result = func(*current_args)
                mapped_result = result_mapper(raw_result)
                analysis_results.update(mapped_result)

                display_payload = None
                main_result_key = list(mapped_result.keys())[0]
                if isinstance(mapped_result[main_result_key], Image.Image if Image else type(None)):
                    display_payload = mapped_result[main_result_key].copy()
                elif isinstance(mapped_result[main_result_key], np.ndarray if np else type(None)):
                    display_payload = mapped_result[main_result_key].copy()
                else: # For dicts, lists, etc.
                    display_payload = mapped_result[main_result_key]

                worker_emitter.step_completed_signal.emit(current_processing_step_num, f"{step_msg_prefix} {name} - {summary_func(raw_result)}", display_payload)
            except Exception as e_func:
                worker_emitter.step_completed_signal.emit(current_processing_step_num, f"{step_msg_prefix} Error in {name}: {e_func}", None)
                try: analysis_results.update({k: None for k in result_mapper({}).keys()}) # Store None for keys
                except: pass

        step_num = base_step_num_for_analysis_loop + len(analysis_step_defs)


        # Fill in missing keys from analysis_functions if any step failed
        # ... (code from previous version for this, slightly adapted if needed) ...

        # 16. Advanced tampering localization
        if not worker_emitter.is_running: return None
        loc_step_msg_prefix = "[16/17]"
        worker_emitter.step_started_signal.emit(step_num, f"{loc_step_msg_prefix} Advanced Tampering Localization")
        if 'advanced_tampering_localization' not in globals() or advanced_tampering_localization is None:
             raise ImportError("Function advanced_tampering_localization is not available.")
        # Pass preprocessed image (could be None if earlier step failed)
        current_preprocessed = preprocessed if 'preprocessed' in locals() and preprocessed is not None else (Image.new("RGB",(100,100)) if Image else None)

        localization_results = advanced_tampering_localization(current_preprocessed, analysis_results, worker_emitter, step_offset=step_num)
        analysis_results['localization_analysis'] = localization_results
        worker_emitter.step_completed_signal.emit(step_num + 2, # It has 3 sub-steps
             f"{loc_step_msg_prefix} Localization done. Tampering: {localization_results.get('tampering_percentage', 0):.1f}%",
             localization_results.get('combined_tampering_mask'))
        step_num += 3

        # 17. Advanced classification
        if not worker_emitter.is_running: return None
        class_step_msg_prefix = "[17/17]"
        worker_emitter.step_started_signal.emit(step_num, f"{class_step_msg_prefix} Manipulation Classification")
        if classify_manipulation_advanced is None:
            raise ImportError("classify_manipulation_advanced is not available.")
        classification = classify_manipulation_advanced(analysis_results)
        analysis_results['classification'] = classification
        worker_emitter.step_completed_signal.emit(step_num, f"{class_step_msg_prefix} Classification: {classification.get('type', 'N/A')}", classification)
        step_num += 1

        processing_time = time.time() - start_time
        analysis_results['processing_time'] = processing_time
        final_msg_prefix = "[🏁]"
        worker_emitter.step_started_signal.emit(step_num, f"{final_msg_prefix} Analysis Complete. Time: {processing_time:.2f}s")
        worker_emitter.step_completed_signal.emit(step_num, f"{final_msg_prefix} Final Result: {classification.get('type', 'N/A')}", analysis_results) # Send all results here

        return analysis_results

    except Exception as e_main:
        if worker_emitter and worker_emitter.is_running:
            import traceback
            error_msg = f"Main analysis error: {str(e_main)}\n{traceback.format_exc()}"
            worker_emitter.analysis_error_signal.emit(error_msg)
        return None


# Main execution for GUI
def main_gui():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    critical_analysis_modules_present = all([
        Image, np, cv2, config, validation,
        perform_multi_quality_ela, extract_multi_detector_features,
    ])
    if not critical_analysis_modules_present:
        print("WARNING: Some core analysis modules (PIL, NumPy, OpenCV, or project specifics) failed to import.")
        print("The GUI will run, but analysis functionality will be severely limited or non-functional.")

    main_gui()
