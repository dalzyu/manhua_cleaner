"""Tests for OCR parsing behavior."""

from pathlib import Path

from PIL import Image

from ..core.ocr import TextSpotter


def test_parse_result_keeps_duplicate_texts(tmp_path: Path) -> None:
    """OCR parsing should not drop regions with duplicate text."""
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (100, 100), color="white").save(img_path)

    raw_result = {
        "spotting_res": {
            "rec_texts": ["hi", "hi"],
            "rec_polys": [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[20, 20], [30, 20], [30, 30], [20, 30]],
            ],
        }
    }

    spotter = TextSpotter()
    regions = spotter._parse_result(raw_result, img_path)

    assert len(regions) == 2
    assert regions[0].shape == (4, 1, 2)
    assert regions[1].shape == (4, 1, 2)
