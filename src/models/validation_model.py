"""
ValidationModel — pure Python model for the Validation pane.

Wraps ValidationState with a typed query/command API.
Views must use this API instead of accessing ValidationState directly.
No Qt dependencies.
"""

from core.validation_state import ValidationState


class ValidationModel:
    def __init__(self, state: ValidationState):
        self.state = state

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #

    def set_poly_path(self, path: str) -> None:
        self.state.poly_path = path

    def set_json_path(self, path: str) -> None:
        self.state.json_path = path

    def set_mask_out_path(self, path: str) -> None:
        self.state.mask_out_path = path
        # Convenience: mask output automatically seeds the GT eval input
        if not self.state.gt_path:
            self.state.gt_path = path

    def set_gt_path(self, path: str) -> None:
        self.state.gt_path = path

    def set_pred_path(self, path: str) -> None:
        self.state.pred_path = path

    def set_eval_out_path(self, path: str) -> None:
        self.state.eval_out_path = path

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_poly_path(self)     -> str: return self.state.poly_path
    def get_json_path(self)     -> str: return self.state.json_path
    def get_mask_out_path(self) -> str: return self.state.mask_out_path
    def get_gt_path(self)       -> str: return self.state.gt_path
    def get_pred_path(self)     -> str: return self.state.pred_path
    def get_eval_out_path(self) -> str: return self.state.eval_out_path

    def can_generate(self) -> bool:
        """True when all Step 1 inputs are set."""
        return bool(
            self.state.poly_path
            and self.state.json_path
            and self.state.mask_out_path
        )

    def can_evaluate(self) -> bool:
        """True when all Step 2 inputs are set."""
        return bool(self.state.gt_path and self.state.pred_path)
