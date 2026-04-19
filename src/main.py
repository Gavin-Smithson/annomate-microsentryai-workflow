import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QInputDialog,
)

from core.utils.logger import setup_logging
setup_logging()

from core.states.dataset_state import DatasetState
from core.states.inference_state import InferenceState
from core.states.validation_state import ValidationState
from models.dataset_model import DatasetTableModel
from models.inference_model import InferenceModel
from models.validation_model import ValidationModel

from controllers.io_controller import IOController
from controllers.inference_controller import InferenceController
from controllers.validation_controller import ValidationController

from views.annomate.window import ImageAnnotator
from views.microsentry.window import MicroSentryWindow
from views.validation.window import ValidationWindow


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnnoMate & MicroSentryAI (MVC)")
        self.resize(1400, 900)

        # Domain state
        self.dataset_state    = DatasetState()
        self.inference_state  = InferenceState()
        self.validation_state = ValidationState()

        # Models
        self.dataset_model    = DatasetTableModel(self.dataset_state)
        self.inference_model  = InferenceModel(self.inference_state)
        self.validation_model = ValidationModel(self.validation_state)

        # Controllers
        self.io_controller         = IOController(self.dataset_model)
        self.inference_controller  = InferenceController(
            self.dataset_model, self.inference_model
        )
        self.validation_controller = ValidationController(self.validation_model)

        # Views
        self.annomate_view    = ImageAnnotator(self.dataset_model, self.io_controller)
        self.sentry_view      = MicroSentryWindow(
            self.dataset_model,
            self.inference_model,
            self.inference_controller,
            self.io_controller,
        )
        self.validation_view  = ValidationWindow(
            self.validation_model, self.validation_controller
        )

        # Cross-tab row sync via each view's public API — no access to internal widgets
        self.annomate_view.row_selection_changed.connect(self.sentry_view.select_row)
        self.sentry_view.row_selection_changed.connect(self.annomate_view.select_row)

        # Polygon transfer: MicroSentry → AnnoMate
        self.sentry_view.polygonsSent.connect(self._handle_polygon_transfer)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.annomate_view,   "AnnoMate")
        self.tabs.addTab(self.sentry_view,     "MicroSentry AI")
        self.tabs.addTab(self.validation_view, "Validation")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)

    def _handle_polygon_transfer(self, polygons: list, default_class: str):
        """Show class-selection dialog then forward polygons to AnnoMate."""
        class_names = self.dataset_model.get_class_names()
        if not class_names:
            class_names = [default_class]

        chosen, ok = QInputDialog.getItem(
            self, "Choose Class", "Assign polygons to class:",
            class_names,
            class_names.index(default_class) if default_class in class_names else 0,
            False,
        )
        if ok and chosen:
            self.annomate_view.receive_polygons(polygons, chosen)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = AppWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
