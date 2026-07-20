from dynlaneseq_eg.tools.make_culane_comparison_figure import (
    ImageResult,
    _choose_failure,
    _choose_largest_gain,
    _choose_representative_gain,
    _f1,
)


def _result(image: str, holistic_f1: float, structured_f1: float, structured_fp: int = 0) -> ImageResult:
    return ImageResult(
        category="curve",
        image=image,
        holistic_tp=0,
        holistic_fp=0,
        holistic_fn=0,
        holistic_f1=holistic_f1,
        structured_tp=0,
        structured_fp=structured_fp,
        structured_fn=0,
        structured_f1=structured_f1,
    )


def test_f1_handles_empty_and_nonempty_cases() -> None:
    assert _f1(0, 0, 0) == 1.0
    assert _f1(2, 1, 1) == 2.0 / 3.0
    assert _f1(0, 2, 3) == 0.0


def test_gain_selection_rules_are_deterministic() -> None:
    values = [
        _result("a.jpg", 0.2, 0.3),
        _result("b.jpg", 0.2, 0.6),
        _result("c.jpg", 0.2, 0.4),
    ]
    assert _choose_representative_gain(values).image == "c.jpg"
    assert _choose_largest_gain(values).image == "b.jpg"


def test_cross_failure_selects_most_false_positives() -> None:
    values = [_result("a.jpg", 1.0, 0.0, 2), _result("b.jpg", 1.0, 0.0, 4)]
    assert _choose_failure(values, "cross").image == "b.jpg"
