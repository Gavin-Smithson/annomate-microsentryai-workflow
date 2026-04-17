"""
ValidationWindow — MVC view for the Validation pane.

QWidget designed for tab embedding.
All dialogs here; all computation delegated to ValidationController.
MVC rule: never access ValidationState directly — use ValidationModel API only.
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QProgressBar, QGroupBox, QScrollArea, QFrame,
)

from models.validation_model import ValidationModel
from controllers.validation_controller import ValidationController

logger = logging.getLogger("Validation.Window")


class ValidationWindow(QWidget):
    """
    Two-step mask validation UI.

    Step 1 — Generate Ground Truth Masks from JSON polygon annotations.
    Step 2 — Run IoU evaluation comparing GT masks against model predictions.
    """

    def __init__(
        self,
        model: ValidationModel,
        controller: ValidationController,
        parent=None,
    ):
        super().__init__(parent)
        self.model      = model
        self.controller = controller
        self._gen_worker  = None
        self._eval_worker = None

        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        root = QVBoxLayout(self)

        # ---- Step 1 ----
        grp_gen = QGroupBox("Step 1: Generate Ground Truth Masks from JSON")
        gen_layout = QVBoxLayout(grp_gen)

        self.lbl_poly, r1 = self._make_row("Select Images:",  self._select_poly)
        self.lbl_json, r2 = self._make_row("Select JSON:",    self._select_json)
        self.lbl_mask_out, r3 = self._make_row(
            "Mask Output:", self._select_mask_out,
            tooltip="Selecting this folder also pre-fills the Step 2 GT path.",
        )
        gen_layout.addLayout(r1)
        gen_layout.addLayout(r2)
        gen_layout.addLayout(r3)

        self.btn_gen = QPushButton("Generate Binary Masks")
        self.btn_gen.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; height: 35px;"
        )
        self.btn_gen.clicked.connect(self._run_generation)
        gen_layout.addWidget(self.btn_gen)
        root.addWidget(grp_gen)

        # ---- Step 2 ----
        grp_eval = QGroupBox("Step 2: Run Evaluation")
        eval_layout = QVBoxLayout(grp_eval)

        self.lbl_gt,   r4 = self._make_row("Select GT Masks:",    self._select_gt)
        self.lbl_pred, r5 = self._make_row("Select Predictions:", self._select_pred)
        eval_layout.addLayout(r4)
        eval_layout.addLayout(r5)

        self.btn_run = QPushButton("Run Comparison")
        self.btn_run.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; height: 35px;"
        )
        self.btn_run.clicked.connect(self._run_evaluation)
        eval_layout.addWidget(self.btn_run)
        root.addWidget(grp_eval)

        # ---- Progress ----
        self.pbar = QProgressBar()
        root.addWidget(self.pbar)

        # ---- Results feed ----
        root.addWidget(QLabel("Evaluation Feed:"))
        self.scroll_area      = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout    = QVBoxLayout(self.results_container)
        self.results_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.results_container)
        root.addWidget(self.scroll_area, stretch=1)

    @staticmethod
    def _make_row(
        button_text: str,
        callback,
        tooltip: str = "",
    ):
        """Return (path_label, QHBoxLayout) for a file-selection row."""
        row = QHBoxLayout()
        btn = QPushButton(button_text)
        btn.setFixedWidth(150)
        if tooltip:
            btn.setToolTip(tooltip)
        btn.clicked.connect(callback)

        lbl = QLabel("Not selected")
        lbl.setStyleSheet("color: gray;")
        lbl.setWordWrap(True)

        row.addWidget(btn)
        row.addWidget(lbl, 1)
        return lbl, row

    # ------------------------------------------------------------------
    # Dialog slots — the only place QFileDialog lives
    # ------------------------------------------------------------------

    def _select_poly(self):
        p = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if p:
            self.model.set_poly_path(p)
            self.lbl_poly.setText(p)
            self.lbl_poly.setStyleSheet("color: black;")

    def _select_json(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select JSON Annotation File", "", "JSON (*.json)"
        )
        if p:
            self.model.set_json_path(p)
            self.lbl_json.setText(p)
            self.lbl_json.setStyleSheet("color: black;")

    def _select_mask_out(self):
        p = QFileDialog.getExistingDirectory(self, "Select Mask Output Folder")
        if p:
            self.model.set_mask_out_path(p)
            self.lbl_mask_out.setText(p)
            self.lbl_mask_out.setStyleSheet("color: black;")
            # model.set_mask_out_path seeds gt_path if it was empty
            gt = self.model.get_gt_path()
            if gt == p:
                self.lbl_gt.setText(p)
                self.lbl_gt.setStyleSheet("color: black;")

    def _select_gt(self):
        p = QFileDialog.getExistingDirectory(self, "Select Ground Truth Masks Folder")
        if p:
            self.model.set_gt_path(p)
            self.lbl_gt.setText(p)
            self.lbl_gt.setStyleSheet("color: black;")

    def _select_pred(self):
        p = QFileDialog.getExistingDirectory(self, "Select Predictions Folder")
        if p:
            self.model.set_pred_path(p)
            self.lbl_pred.setText(p)
            self.lbl_pred.setStyleSheet("color: black;")

    # ------------------------------------------------------------------
    # Worker launch
    # ------------------------------------------------------------------

    def _run_generation(self):
        if not self.model.can_generate():
            QMessageBox.warning(
                self,
                "Missing Inputs",
                "Please select an images folder, a JSON file, and a mask output folder.",
            )
            return

        self._clear_results()
        self._set_ui_state(False)

        try:
            worker = self.controller.start_generation()
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            self._set_ui_state(True)
            return

        worker.progress.connect(self.pbar.setValue)
        worker.log_message.connect(self._add_log_text)
        worker.finished.connect(lambda: self._set_ui_state(True))
        self._gen_worker = worker
        worker.start()

    def _run_evaluation(self):
        if not self.model.can_evaluate():
            QMessageBox.warning(
                self,
                "Missing Inputs",
                "Please select a ground truth folder and a predictions folder.",
            )
            return

        self._clear_results()
        self._set_ui_state(False)

        try:
            worker = self.controller.start_evaluation()
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            self._set_ui_state(True)
            return

        worker.progress.connect(self.pbar.setValue)
        worker.log_message.connect(self._add_log_text)
        worker.match_found.connect(self._add_result_card)
        worker.finished.connect(lambda: self._set_ui_state(True))
        self._eval_worker = worker
        worker.start()

    # ------------------------------------------------------------------
    # Results feed helpers
    # ------------------------------------------------------------------

    def _add_log_text(self, text: str):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #333; font-family: monospace;")
        self.results_layout.addWidget(lbl)
        self._scroll_to_bottom()

    def _add_result_card(self, image_path: str, text: str, iou: float):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        color = "#4CAF50" if iou > 50 else "#F44336"
        card.setStyleSheet(
            f"background-color: white; border: 2px solid {color}; "
            "border-radius: 5px; margin-bottom: 10px;"
        )

        layout = QVBoxLayout(card)

        title = QLabel(text)
        title.setStyleSheet("font-weight: bold; font-size: 14px; border: none;")
        layout.addWidget(title)

        img_lbl = QLabel()
        pix = QPixmap(image_path)
        if not pix.isNull():
            img_lbl.setPixmap(pix.scaledToWidth(400, Qt.SmoothTransformation))
        layout.addWidget(img_lbl)

        self.results_layout.addWidget(card)
        self._scroll_to_bottom()

    def _clear_results(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _scroll_to_bottom(self):
        sb = self.scroll_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_ui_state(self, enabled: bool):
        self.btn_gen.setEnabled(enabled)
        self.btn_run.setEnabled(enabled)
