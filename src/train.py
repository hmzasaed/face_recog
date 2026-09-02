from __future__ import annotations

import csv
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import face_features, load_config, project_path, save_json


def load_split(rows, split):
    X, y = [], []
    for row in rows:
        if row["split"] != split:
            continue
        image = cv2.imread(str(project_path("data") / row["relative_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        X.append(face_features(image))
        y.append(row["participant_id"])
    return np.asarray(X), np.asarray(y)


def main() -> None:
    cfg = load_config()
    rows = list(csv.DictReader(project_path(cfg["paths"]["processed_manifest"]).open("r", encoding="utf-8")))
    X_train, y_train = load_split(rows, "train")
    if len(X_train) == 0 or len(set(y_train)) < 2:
        raise SystemExit("Need training images from at least two participants. Use at least two sessions per person.")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(C=float(cfg["model"]["c_value"]), kernel="linear", probability=True, class_weight="balanced", random_state=int(cfg["random_seed"]))),
    ])
    model.fit(X_train, y_train)
    model_path = project_path(cfg["paths"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_json(project_path(cfg["paths"]["label_map"]), {str(i): label for i, label in enumerate(model.classes_)})
    print(f"Saved model with {len(model.classes_)} classes to {model_path}")


if __name__ == "__main__":
    main()
