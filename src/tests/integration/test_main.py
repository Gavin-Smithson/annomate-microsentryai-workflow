"""
End-to-end smoke tests for the primary Application Window.
Verifies GUI rendering, tab switching, and inter-tab communications.
"""
import pytest
from unittest.mock import patch
from PySide6.QtWidgets import QInputDialog

from main import AppWindow


class TestAppWindowSmoke:
    """High-level tests for application startup and module integration."""
    
    @pytest.fixture
    def main_window(self, qtbot):
        """Creates the main window and registers it with qtbot for teardown."""
        window = AppWindow()
        qtbot.addWidget(window)
        return window

    def test_application_initialization(self, main_window):
        """Verify the main window constructs all MVC components and UI tabs without crashing."""
        # Assert window renders
        assert main_window.windowTitle() == "AnnoMate & MicroSentryAI (MVC)"
        
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

    @patch('main.QInputDialog.getItem')
    def test_cross_tab_polygon_transfer(self, mock_get_item, main_window):
        """Verify AI predictions can be transferred to the manual annotation tab."""
        # Arrange
        # Setup the dataset with a dummy image so it can receive annotations
        dummy_file = "camera_001.jpg"
        main_window.dataset_model.load_folder("/fake/path", [dummy_file])
        main_window.dataset_model.add_class("Defect", (255, 0, 0))
        
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
        annos = main_window.dataset_model.get_annotations(0)
        assert len(annos) == 1, "One polygon annotation should be transferred."
        assert annos[0]["category_name"] == "Defect", "Polygon should be assigned to 'Defect' class."
        assert annos[0]["polygon"] == mock_polygons[0], "Coordinate data must remain intact during transfer."