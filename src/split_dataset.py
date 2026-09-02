from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict

from utils import load_config, project_path


def main() -> None:
    cfg = load_config()
    manifest_path = project_path(cfg["paths"]["processed_manifest"])
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    if not rows:
        raise SystemExit("No processed images found. Run build_and_preprocess.py first.")
    rng = random.Random(int(cfg["random_seed"]))
    by_person = defaultdict(list)
    for row in rows:
        by_person[row["participant_id"]].append(row)
    for person, person_rows in by_person.items():
        sessions = sorted({r["session_id"] for r in person_rows})
        rng.shuffle(sessions)
        if len(sessions) >= 3:
            assignments = {s: "test" if i == 0 else "validation" if i == 1 else "train" for i, s in enumerate(sessions)}
            row_assignments = {}
        elif len(sessions) == 2:
            assignments = {sessions[0]: "test", sessions[1]: "train"}
            row_assignments = {}
        else:
            print(f"Warning: {person} has one session; splitting images may make results optimistic.")
            shuffled_rows = person_rows.copy()
            rng.shuffle(shuffled_rows)
            test_end = max(1, len(shuffled_rows) // 5)
            validation_end = min(len(shuffled_rows), test_end * 2)
            row_assignments = {
                row["image_id"]: "test" if index < test_end else "validation" if index < validation_end else "train"
                for index, row in enumerate(shuffled_rows)
            }
            assignments = {}
        for row in person_rows:
            row["split"] = row_assignments[row["image_id"]] if row_assignments else assignments[row["session_id"]]
            source = project_path("data") / row["relative_path"]
            target = project_path(cfg["paths"]["processed_dir"]) / row["split"] / person / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            row["relative_path"] = str(target.relative_to(project_path("data")))
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Assigned splits and copied processed images into data/processed/{train,validation,test}.")


if __name__ == "__main__":
    main()
