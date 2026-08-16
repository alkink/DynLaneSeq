from __future__ import annotations

from pathlib import Path

from dynlaneseq_eg.tools.convert_condlstr_culane_results import result_relative_path
from dynlaneseq_eg.tools.prepare_condlstr_culane_official import read_population


def test_read_population_keeps_every_entry_and_uses_first_column(tmp_path: Path) -> None:
    source = tmp_path / "train.txt"
    source.write_text("/a/b.jpg /mask.png 1 0\n/a/b.jpg /mask.png 1 0\n/c/d.jpg\n", encoding="utf-8")
    assert read_population(source) == ["a/b.jpg", "a/b.jpg", "c/d.jpg"]


def test_condlstr_comma_name_maps_to_official_relative_path() -> None:
    result = {"image_name": "driver_23_30frame,clip,00020.jpg"}
    assert result_relative_path(result, "unused.jpg") == "driver_23_30frame/clip/00020.jpg"
