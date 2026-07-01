"""CloudCompare plugin for 3DFos."""

import importlib.resources
import typing

import numpy as np
import pycc
from PySide6.QtWidgets import QDialog, QVBoxLayout

from three_d_fos import __version__
from three_d_fos.core.feature import Feature
from three_d_fos.gui.app import MainWidget
from three_d_fos.io import PointCloudData, PointCloudDestination, PointCloudSource, SegmentationResult


class CloudComparePointCloudSource(PointCloudSource):
    """Point cloud source from CC selection."""

    def __init__(self, cc: pycc.ccPythonInstance, point_cloud: pycc.ccPointCloud):
        """Initialize with CC instance, selected point cloud."""
        self.cc = cc
        self.point_cloud = point_cloud
        self._name = point_cloud.getName()

    def get_name(self) -> str:
        """Return the point cloud name."""
        return self._name

    def load(self, features: frozenset[Feature]) -> PointCloudData:
        """Load point cloud data from CloudCompare."""

        # Get coordinates
        xyz = self.point_cloud.points().astype(np.double)

        # Load all requested features
        feature_list = []
        for feat in features:
            # Try to find the scalar field (case-insensitive)
            sf_index = None
            for i in range(self.point_cloud.getNumberOfScalarFields()):
                sf_name = self.point_cloud.getScalarFieldName(i)
                if feat.name.lower() in sf_name.lower():
                    sf_index = i
                    break

            if sf_index is None:
                raise ValueError(f"Point cloud missing required scalar field: '{feat.name}'.")

            feat_data = self.point_cloud.getScalarField(sf_index).toArray()
            if not np.all(np.isfinite(feat_data)):
                raise ValueError(f"Inf values detected in scalar field ({feat.name})")

            # Normalize the feature data
            normalized_data = feat.normalize(feat_data)
            feature_list.append(normalized_data)

        # Combine all features into a single array
        features_array = None
        if feature_list:
            features_array = np.column_stack(feature_list)

        return PointCloudData(xyz=xyz, features=features_array, source_name=self._name)


class CloudComparePointDestination(PointCloudDestination):
    """Save segmentation results back into CC as a new point cloud."""

    def __init__(self, cc: pycc.ccPythonInstance, name: str):
        self.cc = cc
        self._name = name

    def get_name(self) -> str:
        return self._name

    def save(self, result: SegmentationResult) -> None:
        """Create a new point cloud with classification labels and add it to CC."""

        # Create new point cloud from base coordinates
        output_cloud = pycc.ccPointCloud(
            result.original_coord[:, 0], result.original_coord[:, 1], result.original_coord[:, 2]
        )

        # TODO (RJ):
        # output_cloud.copyGlobalShiftAndScale(base_cloud)
        output_cloud.setName(self._name)

        # Add labels as a scalar field
        idx_sf = output_cloud.addScalarField("classification")
        sf_array = output_cloud.getScalarField(idx_sf).asArray()
        sf_array[:] = result.labels.astype(np.float32)[:]
        output_cloud.getScalarField(idx_sf).computeMinAndMax()

        # Set display: show the classification SF with default color scale
        output_cloud.setCurrentDisplayedScalarField(0)
        output_cloud.toggleSF()

        self.cc.addToDB(output_cloud)


class ThreeDFosCC(pycc.PythonPluginInterface):
    """CloudCompare plugin for 3DFos."""

    def __init__(self):
        """Construct the plugin."""
        pycc.PythonPluginInterface.__init__(self)

    @typing.override
    def getIcon(self) -> str:
        """Get the path to the plugin icon."""
        with importlib.resources.as_file(
            importlib.resources.files("three_d_fos.assets") / "3dfos_icon.png"
        ) as icon_path:
            return str(icon_path)

    @typing.override
    def getActions(self) -> list[pycc.Action]:
        """List of actions exposed by the plugin."""
        return [pycc.Action(name="3DFos Segmentation", icon=self.getIcon(), target=main)]


def _create_app_and_run(cc: pycc.ccPythonInstance, point_cloud: pycc.ccPointCloud) -> None:
    """Create and run the 3DFos GUI with CloudCompare point cloud.

    Parameters
    ----------
    cc : pycc.ccPythonInstance
        The CloudCompare instance.
    point_cloud : pycc.ccPointCloud
        The selected point cloud from CloudCompare.
    """
    # Create the main widget first to access its current_backbone
    fos_widget = MainWidget()

    source = CloudComparePointCloudSource(cc, point_cloud)
    destination = CloudComparePointDestination(cc, point_cloud.getName())

    # Override the source selection, use CC cloud as source.
    fos_widget.current_source = source
    fos_widget.current_destination = destination
    fos_widget.source_lbl.setText("Input Cloud")
    fos_widget.source_in.setText(source.get_name())
    fos_widget.source_in.setEnabled(False)
    fos_widget.run_btn.setEnabled(True)
    fos_widget.status_lbl.setText("Ready for inference...")

    # Disable file selection buttosn since we are using CC source
    fos_widget.select_file_btn.setEnabled(False)
    fos_widget.select_file_btn.setVisible(False)
    fos_widget.output_path_in.setVisible(False)
    fos_widget.output_path_lbl.setVisible(False)
    fos_widget.select_output_btn.setEnabled(False)
    fos_widget.select_output_btn.setVisible(False)
    fos_widget.output_path_lbl.setVisible(False)

    # QDialog container
    fos_dialog = QDialog()
    fos_dialog.setWindowTitle(f"3DFos (v{__version__})")
    fos_dialog.setModal(True)
    fos_dialog.setFixedWidth(640)

    # Add a layout to the dialog and widdget to the layout
    dialog_layout = QVBoxLayout(fos_dialog)
    dialog_layout.addWidget(fos_widget)

    fos_dialog.exec()


def main() -> None:
    """3DFos CloudCompare Plugin main action."""
    cc = pycc.GetInstance()

    entities = cc.getSelectedEntities()

    if not entities or len(entities) > 1:
        raise RuntimeError("Please select one point cloud")

    point_cloud = entities[0]

    if not isinstance(point_cloud, pycc.ccPointCloud):
        raise RuntimeError("Selected entity should be a point cloud")

    cc.freezeUI(True)
    try:
        _create_app_and_run(cc, point_cloud)
    except Exception:
        raise RuntimeError("Something went wrong during 3DFos processing!") from None
    finally:
        cc.freezeUI(False)
        cc.updateUI()
