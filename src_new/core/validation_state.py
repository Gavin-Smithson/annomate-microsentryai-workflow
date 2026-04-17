"""
ValidationState — pure Python state container for the Validation pane.

Holds the six directory/file paths needed across the two-step workflow.
Zero Qt dependencies.
"""

import os


class ValidationState:
    def __init__(self):
        self.poly_path:     str = ""
        self.json_path:     str = ""
        self.mask_out_path: str = ""
        self.gt_path:       str = ""
        self.pred_path:     str = ""
        self.eval_out_path: str = os.path.join(os.getcwd(), "evaluation_results")

    def clear(self):
        self.poly_path     = ""
        self.json_path     = ""
        self.mask_out_path = ""
        self.gt_path       = ""
        self.pred_path     = ""
