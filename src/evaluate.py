from __future__ import annotations

import csv
import json

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from utils import face_features, load_config, project_path


def main() -> None:
    cfg = load_config()
    rows = list(csv.DictReader(project_path(cfg["paths"]["processed_manifest"]).open("r", encoding="utf-8")))
    model = joblib.load(project_path(cfg["paths"]["model"]))
    output = {}
    for split in ["validation", "test"]:
        X, y = [], []
        for row in rows:
            if row["split"] != split:
                continue
            image = cv2.imread(str(project_path("data") / row["relative_path"]), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                X.append(face_features(image))
                y.append(row["participant_id"])
        if not X:
            output[split] = {"message": "No images in this split."}
            continue
        X = np.asarray(X)
        y = np.asarray(y)
        pred = model.predict(X)
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        output[split] = {"balanced_accuracy": float(balanced_accuracy_score(y, pred)), "classification_report": report}
        labels = sorted(set(y) | set(pred))
        cm = confusion_matrix(y, pred, labels=labels)
        if split == "test":
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
            fig.tight_layout()
            out = project_path(cfg["paths"]["confusion_matrix"])
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=160)
            plt.close(fig)
    metrics_path = project_path(cfg["paths"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
