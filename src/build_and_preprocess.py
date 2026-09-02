from __future__ import annotations

import csv
from pathlib import Path

import cv2

from utils import detect_single_face, face_features, get_detector, load_config, preprocess_face, project_path


def main() -> None:
    cfg = load_config()
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    processed_dir = project_path(cfg["paths"]["processed_dir"])
    consented = __import__("utils").consented_ids(project_path(cfg["paths"]["consent_log"]))
    size = tuple(cfg["image_size"])
    detector = get_detector()
    rows = []
    for image_path in sorted(raw_dir.rglob("*.jpg")) + sorted(raw_dir.rglob("*.png")):
        relative = image_path.relative_to(raw_dir)
        parts = relative.parts
        if len(parts) < 3:
            continue
        participant_id, session_id = parts[0], parts[1]
        if participant_id not in consented:
            print(f"Skipping {image_path}: no active consent")
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        box = detect_single_face(image, detector)
        if box is None:
            print(f"Skipping {image_path}: expected exactly one detectable face")
            continue
        face = preprocess_face(image, box, size)
        out_dir = processed_dir / participant_id / session_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / image_path.name
        cv2.imwrite(str(out_path), face)
        rows.append({
            "image_id": out_path.stem,
            "relative_path": str(out_path.relative_to(project_path("data"))),
            "participant_id": participant_id,
            "session_id": session_id,
            "split": "unassigned",
            "face_count": 1,
            "width": image.shape[1],
            "height": image.shape[0],
            "consent_confirmed": True,
        })
    manifest = project_path(cfg["paths"]["processed_manifest"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Processed {len(rows)} images. Manifest: {manifest}")


if __name__ == "__main__":
    main()
