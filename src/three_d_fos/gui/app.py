"""PySide6 GUI application for 3DFos."""

import importlib.resources
import typing
from pathlib import Path

import torch
from PySide6.QtCore import QLocale, Qt, QThread, Signal
from PySide6.QtGui import QDoubleValidator, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import three_d_fos
from three_d_fos.core import inference as backend_inference
from three_d_fos.core.model import MODEL_MAP, ModelDefinition
from three_d_fos.io import (
    FilePointCloudDestination,
    FilePointCloudSource,
    PointCloudDestination,
    PointCloudSource,
    SegmentationResult,
)


class InferenceWorker(QThread):
    """Worker thread for running inference to avoid blocking the GUI."""

    progress = Signal(str)
    completed = Signal(SegmentationResult)
    error = Signal(str)

    def __init__(
        self,
        source: PointCloudSource,
        destination: PointCloudDestination,
        device: torch.device,
        grid_size: float,
        model_definition: ModelDefinition,
        tiling_factor: int,
    ):
        super().__init__()
        self.source = source
        self.destination = destination
        self.device = device
        self.grid_size = grid_size
        self.model_definition = model_definition
        self.tiling_factor = tiling_factor

    def run(self) -> None:
        """Run inference on the point cloud and save results."""
        try:
            model = self._get_model(self.model_definition)

            self.progress.emit("Loading point cloud data...")
            data = self.source.load()
            original_coord = data.xyz.copy()

            self.progress.emit("Preprocessing...")
            point_features, remap_ids, _ = backend_inference.preprocess(data, self.grid_size)

            self.progress.emit("Running inference...")
            with torch.no_grad():
                labels = backend_inference.run_inference(model, point_features, self.device, self.tiling_factor)[
                    remap_ids
                ]

            self.progress.emit("Saving results...")
            result = SegmentationResult(original_coord=original_coord, labels=labels)
            self.destination.save(result)

            self.progress.emit("Inference complete!")
            self.completed.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def _get_model(self, model_definition: ModelDefinition) -> torch.nn.Module:
        """Load the segmentation model (in the device)."""
        self.progress.emit("Loading / Downloading model, please wait...")
        try:
            model = three_d_fos.seghead.load(ckpt_path=None, model_definition=model_definition)
            # Move model to target device immediately
            model.to(self.device).eval()
        except Exception as e:
            self.error.emit(f"Failed to load model: {e}")
        self.progress.emit("Model loaded successfully")
        return model


class MainWidget(QWidget):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3DFos - Point Cloud Segmentation")
        self.setGeometry(100, 100, 800, 600)

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.worker: InferenceWorker | None = None
        self.current_source: PointCloudSource | None = None
        self.current_destination: PointCloudDestination | None = None
        self.current_backbone: ModelDefinition | None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)

        # Logo
        logo_label = QLabel()
        with importlib.resources.as_file(
            importlib.resources.files("three_d_fos.assets") / "3DFoS_logo.png"
        ) as logo_path:
            logo_pixmap = QPixmap(str(logo_path))
            logo_label.setPixmap(logo_pixmap.scaledToWidth(200, Qt.SmoothTransformation))
            logo_label.setAlignment(Qt.AlignCenter)
            main_layout.addWidget(logo_label)
            main_layout.addSpacerItem(QSpacerItem(0, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # For now, organize in a grid layout
        parameters_layout = QGridLayout()

        self.device_lbl = QLabel("Device")
        # Populate with available devices
        devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        if len(devices) == 1:
            self.device_in = QLabel(devices[0])
            self.device = torch.device(devices[0])
        else:
            self.device_in = QComboBox()
            self.device_in.addItems(devices)
            self.device_in.setCurrentText(self.device.type)
            self.device_in.currentTextChanged.connect(self._on_device_changed)

        self.device_in.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        parameters_layout.addWidget(self.device_lbl, 0, 0)
        parameters_layout.addWidget(self.device_in, 0, 1)

        # Backbone
        self.backbone_lbl = QLabel("Backbone")
        self.backbone_in = QComboBox()
        self.backbone_in.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.backbone_in.addItems(list(MODEL_MAP.keys()))
        self.backbone_in.currentTextChanged.connect(self._on_backbone_changed)
        parameters_layout.addWidget(self.backbone_lbl, 1, 0)
        parameters_layout.addWidget(self.backbone_in, 1, 1)

        # Voxel size
        self.grid_size_lbl = QLabel("Voxel size")
        self.grid_size_in = QLineEdit("0.05")
        self.grid_size_in.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        voxel_validator = QDoubleValidator(0.001, 5.0, 4)
        voxel_validator.setLocale(QLocale.c())
        self.grid_size_in.setValidator(voxel_validator)

        parameters_layout.addWidget(self.grid_size_lbl, 2, 0)
        parameters_layout.addWidget(self.grid_size_in, 2, 1)

        # Tiling factor
        self.tiling_factor_lbl = QLabel("Tiling factor")
        self.tiling_factor_in = QLineEdit("0")
        self.tiling_factor_in.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        tiling_validator = QIntValidator(0, 10)
        self.tiling_factor_in.setValidator(tiling_validator)
        parameters_layout.addWidget(self.tiling_factor_lbl, 3, 0)
        parameters_layout.addWidget(self.tiling_factor_in, 3, 1)

        # Input selection
        self.source_lbl = QLabel("Input path")
        self.source_in = QLineEdit()
        self.source_in.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.select_file_btn = QPushButton("Open file")
        self.select_file_btn.clicked.connect(self._on_select_file)
        parameters_layout.addWidget(self.source_lbl, 4, 0)
        parameters_layout.addWidget(self.source_in, 4, 1)
        parameters_layout.addWidget(self.select_file_btn, 4, 2)

        # Output selection
        self.output_path_lbl = QLabel("Output path")
        self.output_path_in = QLineEdit()
        self.output_path_in.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.select_output_btn = QPushButton("Select path")
        self.select_output_btn.clicked.connect(self._on_select_output)
        parameters_layout.addWidget(self.output_path_lbl, 5, 0)
        parameters_layout.addWidget(self.output_path_in, 5, 1)
        parameters_layout.addWidget(self.select_output_btn, 5, 2)

        main_layout.addLayout(parameters_layout)

        # Run button
        button_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run")
        self.run_btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self.run_btn.clicked.connect(self._on_run_inference)
        self.run_btn.setEnabled(False)
        button_layout.addStretch()
        button_layout.addWidget(self.run_btn)
        main_layout.addLayout(button_layout)

        # Status
        self.status_bar = QStatusBar(self)
        self.status_lbl = QLabel("Please select an input file")
        self.status_bar.addWidget(self.status_lbl)
        main_layout.addWidget(self.status_bar)

    def _on_device_changed(self, device_str: str) -> None:
        """Handle device selection."""
        self.device = torch.device(device_str)

    def _on_backbone_changed(self, backbone_str: str) -> None:
        """Handle backbone selection"""
        self.current_backbone = MODEL_MAP[backbone_str.lower()]

    def _on_select_file(self) -> None:
        """Handle file selection."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Point Cloud File",
            "",
            "Point Cloud Files (*.ply *.las *.laz);;All Files (*)",
        )

        if filepath:
            self.current_source = FilePointCloudSource(Path(filepath), self.current_backbone.features)
            self.source_in.setText(self.current_source.get_name())
            self.run_btn.setEnabled(True)
            self.status_lbl.setText("Source loaded, ready for inference")

    def _on_select_output(self) -> None:
        """Handle output path selection."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output LAS File",
            "seg_result.las",
            "LAS Files (*.las);;All Files (*)",
        )

        if filepath:
            self.output_path_in.setText(filepath)
            self.current_destination = FilePointCloudDestination(Path(filepath))

    def _on_run_inference(self) -> None:
        """Handle inference execution."""
        if not self.current_source:
            QMessageBox.warning(self, "Warning", "No source selected!")
            return

        if not self.current_destination:
            QMessageBox.warning(self, "Warning", "No output path selected!")
            return

        # Disable UI during processing
        self.run_btn.setEnabled(False)
        self.select_file_btn.setEnabled(False)
        self.select_output_btn.setEnabled(False)
        self.status_lbl.setText("Processing...")

        # Create and start worker thread
        self.worker = InferenceWorker(
            source=self.current_source,
            destination=self.current_destination,
            device=self.device,
            grid_size=float(self.grid_size_in.text()),
            model_definition=self.current_backbone,
            tiling_factor=int(self.tiling_factor_in.text()),
        )

        self.worker.progress.connect(self.status_lbl.setText)
        self.worker.completed.connect(self._on_inference_complete)
        self.worker.error.connect(self._on_inference_error)
        self.worker.start()

    def _on_inference_complete(self, result: SegmentationResult) -> None:
        """Handle inference completion."""
        self.status_lbl.setText("Inference done!")
        self.run_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.select_output_btn.setEnabled(True)

        QMessageBox.information(self, "Success", f"Results saved to {self.current_destination.get_name()}")

    def _on_inference_error(self, error_msg: str) -> None:
        """Handle inference error."""
        self.status_lbl.setText(f"Error: {error_msg}")
        self.run_btn.setEnabled(True)
        self.select_file_btn.setEnabled(True)
        self.select_output_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", error_msg)

    @typing.override
    def closeEvent(self, event) -> None:
        """Handle window close."""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()


class ThreeDFosApp:
    """Main application class for 3DFos (GUI)."""

    def __init__(self):
        self.app = QApplication([])
        self.app.setApplicationName("3DFos")
        self._set_window_icon()
        fos_widget = MainWidget()
        self.window = QMainWindow()
        self.window.setWindowTitle("3DFos")
        self.window.setCentralWidget(fos_widget)
        self.window.setFixedWidth(640)

    def _set_window_icon(self) -> None:
        """Set the window icon."""
        with importlib.resources.as_file(
            importlib.resources.files("three_d_fos.assets") / "3dfos_icon.png"
        ) as icon_path:
            self.app.setWindowIcon(QPixmap(str(icon_path)))

    def run(self) -> None:
        """Run the application."""
        self.window.show()
        self.app.exec()


def main():
    """Entry point for GUI application."""
    app = ThreeDFosApp()
    app.run()


if __name__ == "__main__":
    main()
