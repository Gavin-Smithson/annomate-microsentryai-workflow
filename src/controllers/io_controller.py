"""
IOController — headless business logic for file I/O.

Rules:
  - No Qt GUI types (no QFileDialog, QMessageBox, QColor).
  - All methods accept plain Python values (paths, strings).
  - Errors are signalled by raising exceptions; callers (Views) handle display.
"""

import os
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.utils.constants import DEFAULT_CLASS_COLORS

logger = logging.getLogger("AnnoMate.IOController")


class IOController:
    def __init__(self, model):
        self.model = model

    def load_folder(self, directory: str):
        """Scan a directory for images and load them into the model."""
        logger.debug("Scanning directory for images: %s", directory)
        
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted(
            f for f in os.listdir(directory) if Path(f).suffix.lower() in exts
        )
        
        logger.debug("Found %d valid images in folder.", len(files))
        self.model.load_folder(directory, files)

    def load_image_for_display(self, row: int) -> Optional[np.ndarray]:
        """
        Read the image at *row* from disk and return a BGR ndarray.
        Returns None if the file cannot be read.
        """
        path = self.model.get_image_path(row)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning(f"Could not read image: {path}")
        return bgr

    def export_polygons_and_data(self, out_dir: str) -> str:
        """
        Write overlay images + a JSON data file to *out_dir*.
        Returns a human-readable success message.
        Raises RuntimeError on I/O failure.
        """
        state = self.model.state
        if not state.image_files:
            logger.warning("Attempted to export, but no images are loaded.")
            raise RuntimeError("No images loaded.")

        logger.debug("Starting polygon export to: %s", out_dir)
        
        out_path = Path(out_dir)
        tray_name = Path(state.image_dir).name if state.image_dir else "tray"
        timestamp = datetime.now().strftime("%m-%d-%y-%H-%M-%S")

        payload = {
            "meta": {"tray": tray_name, "exported_at": timestamp},
            "classes": list(state.class_names),
            # Colors stored as (r,g,b) tuples — JSON-serialisable, no Qt needed.
            "class_colors": {
                name: list(rgb) for name, rgb in state.class_colors.items()
            },
            "images": {},
        }

        saved_count = 0
        for name in state.image_files:
            anns = state.annotations.get(name, [])
            is_rev = state.is_reviewed(name)

            payload["images"][name] = {
                "inspector": state.inspectors.get(name, "") if is_rev else "",
                "note": state.notes.get(name, "") if is_rev else "",
                "annotations": [
                    {
                        "class": a["category_name"],
                        "polygon": [(float(x), float(y)) for (x, y) in a["polygon"]],
                    }
                    for a in anns
                ],
            }

            if not anns:
                continue

            src = Path(state.image_dir) / name
            if not src.exists():
                continue

            try:
                base = Image.open(src).convert("RGBA")
                overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay, "RGBA")

                for a in anns:
                    pts = [(float(x), float(y)) for (x, y) in a["polygon"]]
                    if len(pts) < 2:
                        continue
                    rgb = state.class_colors.get(a["category_name"], (255, 255, 255))
                    draw.polygon(pts, fill=(*rgb, 80), outline=(*rgb, 255))
                    draw.line(pts + [pts[0]], fill=(*rgb, 255), width=3)

                composed = Image.alpha_composite(base, overlay).convert("RGB")
                out_name = f"{tray_name}_{Path(name).stem}_{timestamp}_poly.jpg"
                composed.save(out_path / out_name, "JPEG", quality=95)
                saved_count += 1
            except Exception as e:
                logger.error(f"Failed to export overlay for {name}: {e}")

        data_path = out_path / f"{tray_name}_{timestamp}_data.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        logger.debug("Successfully exported %d overlay images and data JSON.", saved_count)
        return f"Saved {saved_count} image(s) + data JSON:\n{data_path}"

    def export_csv(self, out_path: str) -> str:
        """
        Write per-image metadata to a CSV at *out_path*.
        Returns a success message. Raises on I/O failure.
        """
        state = self.model.state
        if not state.image_files:
            raise RuntimeError("No images loaded.")

        tray_name = Path(state.image_dir).name if state.image_dir else ""
        rows = []
        for name in state.image_files:
            anns = state.annotations.get(name, [])
            unique_classes = sorted({a["category_name"] for a in anns})
            reviewed = state.is_reviewed(name)
            rows.append(
                {
                    "tray": tray_name,
                    "image_name": name,
                    "inspector": state.inspectors.get(name, "") if reviewed else "",
                    "note": state.notes.get(name, "") if reviewed else "",
                    "classes": (
                        ", ".join(unique_classes) if unique_classes else "good"
                    ) if reviewed else "",
                }
            )

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["tray", "image_name", "inspector", "note", "classes"]
            )
            writer.writeheader()
            writer.writerows(rows)

        return f"CSV saved to:\n{out_path}"

    # ------------------------------------------------------------------ #
    # Import
    # ------------------------------------------------------------------ #

    def import_data_json(self, path: str):
        """
        Load annotations from a custom or COCO JSON file into the model.
        Raises on parse/import errors.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state = self.model.state
        state.annotations.clear()
        state.inspectors.clear()
        state.notes.clear()

        images_node = data.get("images")
        if isinstance(images_node, dict):
            self._import_custom_format(state, data, images_node)
        elif isinstance(images_node, list):
            self._import_coco_format(state, data, images_node)

        # Tell all attached views to fully refresh
        self.model.beginResetModel()
        self.model.endResetModel()

    def _import_custom_format(self, state, data: dict, images_node: dict):
        classes = data.get("classes", [])
        if classes:
            state.class_names = list(classes)
            saved_colors = data.get("class_colors", {})
            state.class_colors = {}
            for i, name in enumerate(state.class_names):
                raw = saved_colors.get(name)
                if isinstance(raw, (list, tuple)) and len(raw) == 3:
                    state.class_colors[name] = (int(raw[0]), int(raw[1]), int(raw[2]))
                else:
                    state.class_colors[name] = DEFAULT_CLASS_COLORS[i % len(DEFAULT_CLASS_COLORS)]

        for name, info in images_node.items():
            state.inspectors[name] = info.get("inspector", "")
            state.notes[name] = info.get("note", "")
            recs = [
                {"category_name": a.get("class", ""), "polygon": a.get("polygon", [])}
                for a in info.get("annotations", [])
            ]
            if recs:
                state.annotations[name] = recs

    def _import_coco_format(self, state, data: dict, images_node: list):
        cat_map = {}
        if "categories" in data:
            for c in data["categories"]:
                name = c["name"]
                cat_map[c["id"]] = name
                if name not in state.class_names:
                    idx = len(state.class_names)
                    state.class_names.append(name)
                    state.class_colors[name] = DEFAULT_CLASS_COLORS[idx % len(DEFAULT_CLASS_COLORS)]

        img_id_map = {img["id"]: img["file_name"] for img in images_node}

        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in img_id_map:
                continue
            filename = img_id_map[img_id]
            cat_name = cat_map.get(ann["category_id"], "Unknown")
            seg = ann.get("segmentation", [])
            final_poly = []
            if isinstance(seg, list) and seg:
                pts_list = seg[0] if isinstance(seg[0], list) else seg
                for i in range(0, len(pts_list) - 1, 2):
                    final_poly.append((float(pts_list[i]), float(pts_list[i + 1])))
            if final_poly:
                state.annotations.setdefault(filename, []).append(
                    {"category_name": cat_name, "polygon": final_poly}
                )
