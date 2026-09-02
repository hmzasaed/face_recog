from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from skimage.feature import hog

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict[str, Any]:
    with (PROJECT_ROOT / "configs" / "config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str | Path) -> Path:
    return PROJECT_ROOT / Path(relative)


def get_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load face detector: {cascade_path}")
    return detector


def detect_single_face(image: np.ndarray, detector: cv2.CascadeClassifier) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) != 1:
        return None
    return tuple(int(v) for v in faces[0])


def preprocess_face(image: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    x, y, w, h = box
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    crop = gray[y:y + h, x:x + w]
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(crop)


def face_features(face: np.ndarray) -> np.ndarray:
    values = hog(
        face,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return values.astype(np.float32)


def consented_ids(consent_path: Path) -> set[str]:
    if not consent_path.exists():
        return set()
    import csv
    with consent_path.open("r", encoding="utf-8", newline="") as f:
        return {
            row["participant_id"].strip()
            for row in csv.DictReader(f)
            if row.get("consent_confirmed", "").strip().lower() == "true"
            and row.get("withdrawn", "").strip().lower() != "true"
        }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
