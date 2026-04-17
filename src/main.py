import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QInputDialog,
)

from core.logger import setup_logging
setup_logging()

from core.dataset_state import DatasetState
from core.inference_state import InferenceState
from core.validation_state import ValidationState
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
            self.dataset_model, self.inference_model, self.inference_controller
        )
        self.validation_view  = ValidationWindow(
            self.validation_model, self.validation_controller
        )

        # Cross-tab row sync via selection signals (proxy models differ, so we
        # use explicit signal connections rather than a shared QItemSelectionModel)
        annomate_sel = self.annomate_view.table_view.selectionModel()
        sentry_sel   = self.sentry_view.table_view.selectionModel()
        annomate_sel.currentRowChanged.connect(
            lambda c, _: sentry_sel.setCurrentIndex(
                self.sentry_view._proxy.index(c.row(), 0),
                sentry_sel.ClearAndSelect | sentry_sel.Rows,
            )
        )
        sentry_sel.currentRowChanged.connect(
            lambda c, _: annomate_sel.setCurrentIndex(
                self.dataset_model.index(c.row(), 0),
                annomate_sel.ClearAndSelect | annomate_sel.Rows,
            )
        )

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