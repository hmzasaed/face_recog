from __future__ import annotations

import csv
import time
from datetime import date
from pathlib import Path

import cv2

from utils import PROJECT_ROOT, load_config, project_path


def main() -> None:
    cfg = load_config()
    participant_id = input("Pseudonymous participant ID (for example person_01): ").strip()
    session_id = input("Session ID (for example session_01): ").strip()
    consent = input("Has this participant explicitly consented? Type YES to continue: ").strip()
    if consent != "YES":
        raise SystemExit("Collection cancelled: explicit consent is required.")
    if not participant_id or not session_id or any(ch in participant_id + session_id for ch in " /\\"):
        raise SystemExit("Use non-empty IDs without spaces or path separators.")

    consent_path = project_path(cfg["paths"]["consent_log"])
    consent_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not consent_path.exists()
    with consent_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_id", "consent_date", "consent_confirmed", "withdrawn"])
        if new_file:
            writer.writeheader()
        writer.writerow({"participant_id": participant_id, "consent_date": date.today().isoformat(), "consent_confirmed": "true", "withdrawn": "false"})

    output_dir = project_path(cfg["paths"]["raw_dir"]) / participant_id / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(int(cfg["collection"]["camera_index"]))
    if not camera.isOpened():
        raise SystemExit("Could not open the webcam. Check camera permissions and camera_index.")

    target = int(cfg["collection"]["target_images_per_session"])
    interval = float(cfg["collection"]["capture_interval_seconds"])
    saved = 0
    last_save = 0.0
    print("Look at the camera. Press q to cancel. Images are saved locally only.")
    try:
        while saved < target:
            ok, frame = camera.read()
            if not ok:
                break
            display = frame.copy()
            cv2.putText(display, f"Saved: {saved}/{target} | q=quit", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Consented dataset collection", display)
            now = time.time()
            if now - last_save >= interval:
                filename = output_dir / f"img_{saved + 1:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                saved += 1
                last_save = now
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} images to {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
