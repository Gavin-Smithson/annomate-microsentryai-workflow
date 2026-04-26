"""
End-to-end smoke tests for the primary Application Window.
Verifies GUI rendering, tab switching, and inter-tab communications.
"""

import pytest
from unittest.mock import patch, MagicMock

from views.app_window import AppWindow
from core.states.dataset_state import DatasetState
from core.states.inference_state import InferenceState
from core.states.validation_state import ValidationState
from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from models.validation_model import ValidationModel
from controllers.io_controller import IOController
from controllers.inference_controller import InferenceController
from controllers.validation_controller import ValidationController


class TestAppWindowSmoke:
    """High-level tests for application startup and module integration."""

    @pytest.fixture
    def injected_components(self):
        """Instantiates the core MVC components required by the new AppWindow DI architecture."""
        dataset_model = DatasetTableModel(DatasetState())
        inference_model = InferenceModel(InferenceState())
        validation_model = ValidationModel(ValidationState())

        io_controller = IOController(dataset_model)
        inference_controller = InferenceController(dataset_model, inference_model)

        # Mock the ML strategy to prevent PyTorch from loading in the smoke test
        inference_controller._strategy = MagicMock()

        validation_controller = ValidationController(validation_model)

        # Mock the new ProjectController to keep the smoke test focused purely
        # on UI integration and avoid potential side-effects
        project_controller = MagicMock()
        # Force the mock to report no unsaved changes just in case
        project_controller.has_unsaved_changes.return_value = False

        # Pack into a kwargs dictionary for clean injection
        return {
            "dataset_model": dataset_model,
            "inference_model": inference_model,
            "validation_model": validation_model,
            "io_controller": io_controller,
            "inference_controller": inference_controller,
            "validation_controller": validation_controller,
            "project_controller": project_controller,
        }

    @pytest.fixture
    def main_window(self, qtbot, injected_components):
        """Creates the main window using dependency injection and registers it with qtbot."""
        window = AppWindow(**injected_components)

        # CRITICAL FIX: Bypass the closeEvent to prevent "Save Changes?" dialogs
        # from blocking the Qt event loop during test teardown.
        window.closeEvent = lambda event: event.accept()

        qtbot.addWidget(window)
        return window

    def test_application_initialization(self, main_window):
        """Verify the main window constructs all UI tabs without crashing."""
        # Assert window renders (Updated to match the new merged title)
        assert main_window.windowTitle() == "AnnoMate & MicroSentryAI"

        # Assert tabs exist
        assert main_window.tabs.count() == 3
        assert main_window.tabs.tabText(0) == "AnnoMate"
        assert main_window.tabs.tabText(1) == "MicroSentry AI"
        assert main_window.tabs.tabText(2) == "Validation"

    def test_tab_switching(self, main_window):
        """Verify user can safely navigate between the core modes."""
        # Arrange & Act: Switch to MicroSentry
        main_window.tabs.setCurrentIndex(1)
        assert main_window.tabs.currentIndex() == 1

        # Arrange & Act: Switch to Validation
        main_window.tabs.setCurrentIndex(2)
        assert main_window.tabs.currentIndex() == 2

    @patch("views.app_window.QInputDialog.getItem")
    def test_cross_tab_polygon_transfer(
        self, mock_get_item, main_window, injected_components
    ):
        """Verify AI predictions can be transferred to the manual annotation tab."""
        # Arrange
        dataset_model = injected_components["dataset_model"]

        # Setup the dataset with a dummy image so it can receive annotations
        dummy_file = "camera_001.jpg"
        dataset_model.load_folder("/fake/path", [dummy_file])
        dataset_model.add_class("Defect", (255, 0, 0))

        # Select the first row to make it active
        main_window.annomate_view.select_row(0)

        # Mock the QInputDialog to simulate user clicking "OK" and selecting "Defect"
        mock_get_item.return_value = ("Defect", True)

        mock_polygons = [[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0)]]

        # Act
        # Simulate MicroSentry emitting AI polygons
        main_window.sentry_view.polygonsSent.emit(mock_polygons, "Defect")

        # Assert
        # The annotation should now exist inside the AnnoMate dataset state
        annos = dataset_model.get_annotations(0)
        assert len(annos) == 1, "One polygon annotation should be transferred."
        assert (
            annos[0]["category_name"] == "Defect"
        ), "Polygon should be assigned to 'Defect' class."
        assert (
            annos[0]["polygon"] == mock_polygons[0]
        ), "Coordinate data must remain intact during transfer."
