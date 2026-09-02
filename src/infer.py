from __future__ import annotations

import argparse
import time

import cv2
import joblib

from utils import detect_single_face, face_features, get_detector, load_config, preprocess_face, project_path


def predict(frame, model, detector, size, threshold):
    box = detect_single_face(frame, detector)
    if box is None:
        return "No single face", 0.0, None
    face = preprocess_face(frame, box, size)
    probabilities = model.predict_proba([face_features(face)])[0]
    index = int(probabilities.argmax())
    confidence = float(probabilities[index])
    label = str(model.classes_[index]) if confidence >= threshold else "unknown"
    return label, confidence, box


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to one consented test image")
    args = parser.parse_args()
    cfg = load_config()
    model = joblib.load(project_path(cfg["paths"]["model"]))
    detector = get_detector()
    size = tuple(cfg["image_size"])
    threshold = float(cfg["model"]["unknown_probability_threshold"])
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit("Could not read image.")
        label, confidence, _ = predict(frame, model, detector, size, threshold)
        print(f"Prediction: {label} | confidence: {confidence:.3f}")
        return
    camera = cv2.VideoCapture(int(cfg["collection"]["camera_index"]))
    if not camera.isOpened():
        raise SystemExit("Could not open webcam.")
    print("Webcam running. Press q to quit. Frames are not saved.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            label, confidence, box = predict(frame, model, detector, size, threshold)
            if box:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Face identification", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
