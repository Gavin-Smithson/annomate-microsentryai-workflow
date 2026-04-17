from core.constants import DEFAULT_CLASSES


class DatasetState:
    def __init__(self):
        # File management
        self.image_dir = ""
        self.image_files = []

        # Annotations & Metadata
        self.annotations = {}   # { "img.jpg": [ { "category_name": str, "polygon": [...] } ] }
        self.inspectors = {}    # { "img.jpg": "John Doe" }
        self.notes = {}         # { "img.jpg": "Needs review" }

        # Class registry — initialized from defaults, NOT cleared on folder load
        self.class_names = list(DEFAULT_CLASSES.keys())
        self.class_colors = dict(DEFAULT_CLASSES)   # { name: (r, g, b) }

    def clear(self):
        """Reset per-folder data. Class registry is intentionally preserved."""
        self.image_dir = ""
        self.image_files = []
        self.annotations.clear()
        self.inspectors.clear()
        self.notes.clear()

    def is_reviewed(self, img_name: str) -> bool:
        has_anno = bool(self.annotations.get(img_name))
        has_meta = bool(self.inspectors.get(img_name) or self.notes.get(img_name))
        return has_anno or has_meta

    # --- Annotation CRUD ---

    def add_annotation(self, image_name: str, category: str, polygon: list):
        self.annotations.setdefault(image_name, []).append(
            {"category_name": category, "polygon": polygon}
        )

    def delete_annotation(self, image_name: str, index: int):
        annos = self.annotations.get(image_name, [])
        if 0 <= index < len(annos):
            annos.pop(index)

    def update_annotation_points(self, image_name: str, index: int, points: list):
        annos = self.annotations.get(image_name, [])
        if 0 <= index < len(annos):
            annos[index]["polygon"] = points

    # --- Class Registry ---

    def add_class(self, name: str, color):
        if name not in self.class_names:
            self.class_names.append(name)
            self.class_colors[name] = color

    def delete_class(self, name: str):
        if name in self.class_names:
            self.class_names.remove(name)
            self.class_colors.pop(name, None)
            for img in self.annotations:
                self.annotations[img] = [
                    a for a in self.annotations[img]
                    if a.get("category_name") != name
                ]

    # --- Per-image Metadata ---

    def set_inspector(self, image_name: str, value: str):
        self.inspectors[image_name] = value

    def set_note(self, image_name: str, value: str):
        self.notes[image_name] = value
