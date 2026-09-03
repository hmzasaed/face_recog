from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_and_preprocess import resolve_raw_image_metadata


def test_resolve_raw_image_metadata_supports_direct_files_and_sessions(tmp_path):
    direct = tmp_path / "Akshay Kumar" / "Akshay Kumar_1.jpg"
    direct.parent.mkdir(parents=True, exist_ok=True)
    direct.write_bytes(b"fake-image")

    assert resolve_raw_image_metadata(direct, tmp_path) == ("Akshay Kumar", "session_01")

    nested = tmp_path / "Bilal_Mukhtar" / "Session_22" / "img_0001.jpg"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(b"fake-image")

    assert resolve_raw_image_metadata(nested, tmp_path) == ("Bilal_Mukhtar", "Session_22")
