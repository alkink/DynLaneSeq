from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from dynlaneseq_eg.evaluation.culane_metric import culane_metric, list_image_rel_paths, load_culane_img_data


CATEGORY_LISTS = {
    "normal": "dataset/list/test_split/test0_normal.txt",
    "crowd": "dataset/list/test_split/test1_crowd.txt",
    "hlight": "dataset/list/test_split/test2_hlight.txt",
    "shadow": "dataset/list/test_split/test3_shadow.txt",
    "noline": "dataset/list/test_split/test4_noline.txt",
    "arrow": "dataset/list/test_split/test5_arrow.txt",
    "curve": "dataset/list/test_split/test6_curve.txt",
    "cross": "dataset/list/test_split/test7_cross.txt",
    "night": "dataset/list/test_split/test8_night.txt",
}

GT_COLOR = (75, 235, 110)
HOLISTIC_COLOR = (255, 158, 44)
STRUCTURED_COLOR = (30, 210, 255)


@dataclass(frozen=True)
class ImageResult:
    category: str
    image: str
    holistic_tp: int
    holistic_fp: int
    holistic_fn: int
    holistic_f1: float
    structured_tp: int
    structured_fp: int
    structured_fn: int
    structured_f1: float

    @property
    def delta_f1(self) -> float:
        return self.structured_f1 - self.holistic_f1


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2.0 * tp / denominator


def _evaluate_one(
    item: tuple[str, str],
    *,
    data_root: str,
    holistic_pred_dir: str,
    structured_pred_dir: str,
    width: int,
    iou_threshold: float,
) -> ImageResult:
    category, rel = item
    line_rel = str(Path(rel).with_suffix(".lines.txt"))
    gt = load_culane_img_data(Path(data_root) / line_rel)
    holistic = load_culane_img_data(Path(holistic_pred_dir) / line_rel)
    structured = load_culane_img_data(Path(structured_pred_dir) / line_rel)
    h_tp, h_fp, h_fn = culane_metric(
        holistic, gt, width=width, iou_thresholds=(iou_threshold,), official=True
    )[iou_threshold]
    s_tp, s_fp, s_fn = culane_metric(
        structured, gt, width=width, iou_thresholds=(iou_threshold,), official=True
    )[iou_threshold]
    return ImageResult(
        category=category,
        image=rel,
        holistic_tp=h_tp,
        holistic_fp=h_fp,
        holistic_fn=h_fn,
        holistic_f1=_f1(h_tp, h_fp, h_fn),
        structured_tp=s_tp,
        structured_fp=s_fp,
        structured_fn=s_fn,
        structured_f1=_f1(s_tp, s_fp, s_fn),
    )


def _choose_representative_gain(results: list[ImageResult]) -> ImageResult:
    positive = sorted((result for result in results if result.delta_f1 > 0.0), key=lambda x: (x.delta_f1, x.image))
    pool = positive if positive else sorted(results, key=lambda x: (x.delta_f1, x.image))
    return pool[(len(pool) - 1) // 2]


def _choose_largest_gain(results: list[ImageResult]) -> ImageResult:
    return max(results, key=lambda x: (x.delta_f1, x.structured_f1, x.image))


def _choose_failure(results: list[ImageResult], category: str) -> ImageResult:
    if category == "cross":
        return max(results, key=lambda x: (x.structured_fp, x.holistic_fp, x.image))
    return min(results, key=lambda x: (x.structured_f1, -x.structured_fn, -x.structured_fp, x.image))


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:  # pragma: no cover - depends on host fonts
        return ImageFont.load_default()


def _draw_lanes(image: Image.Image, lanes: Iterable[Iterable[tuple[float, float]]], color, width: int) -> None:
    draw = ImageDraw.Draw(image)
    for lane in lanes:
        points = [(round(float(x)), round(float(y))) for x, y in lane]
        if len(points) >= 2:
            draw.line(points, fill=color, width=width, joint="curve")


def _panel(
    image_path: Path,
    gt_path: Path,
    pred_path: Path | None,
    *,
    pred_color,
    title: str,
    metrics: str,
    panel_width: int,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    gt = load_culane_img_data(gt_path)
    _draw_lanes(image, gt, GT_COLOR, width=4)
    if pred_path is not None:
        _draw_lanes(image, load_culane_img_data(pred_path), pred_color, width=5)
    panel_height = round(panel_width * image.height / image.width)
    image = image.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
    bar_height = 44
    panel = Image.new("RGB", (panel_width, panel_height + bar_height), (22, 25, 31))
    panel.paste(image, (0, bar_height))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 4), title, fill=(248, 248, 248), font=_font(16, bold=True))
    draw.text((10, 24), metrics, fill=(205, 211, 222), font=_font(13))
    return panel


def _make_montage(
    selections: list[tuple[str, ImageResult]],
    *,
    data_root: Path,
    holistic_pred_dir: Path,
    structured_pred_dir: Path,
    output_path: Path,
    title: str,
    panel_width: int,
) -> None:
    rows: list[Image.Image] = []
    for rule, result in selections:
        rel = Path(result.image)
        line_rel = rel.with_suffix(".lines.txt")
        image_path = data_root / rel
        gt_path = data_root / line_rel
        gt_panel = _panel(
            image_path,
            gt_path,
            None,
            pred_color=GT_COLOR,
            title=f"{result.category.upper()} | Ground truth",
            metrics=f"selection: {rule}",
            panel_width=panel_width,
        )
        holistic_panel = _panel(
            image_path,
            gt_path,
            holistic_pred_dir / line_rel,
            pred_color=HOLISTIC_COLOR,
            title="Holistic query",
            metrics=(
                f"TP/FP/FN {result.holistic_tp}/{result.holistic_fp}/{result.holistic_fn}  "
                f"F1 {result.holistic_f1:.3f}"
            ),
            panel_width=panel_width,
        )
        structured_panel = _panel(
            image_path,
            gt_path,
            structured_pred_dir / line_rel,
            pred_color=STRUCTURED_COLOR,
            title="Structured row query",
            metrics=(
                f"TP/FP/FN {result.structured_tp}/{result.structured_fp}/{result.structured_fn}  "
                f"F1 {result.structured_f1:.3f}  delta {result.delta_f1:+.3f}"
            ),
            panel_width=panel_width,
        )
        row = Image.new("RGB", (panel_width * 3, gt_panel.height), (255, 255, 255))
        row.paste(gt_panel, (0, 0))
        row.paste(holistic_panel, (panel_width, 0))
        row.paste(structured_panel, (panel_width * 2, 0))
        rows.append(row)

    header_height = 54
    canvas = Image.new(
        "RGB",
        (panel_width * 3, header_height + sum(row.height for row in rows)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(20, 24, 30), font=_font(23, bold=True))
    draw.text(
        (12, 34),
        "GT: green | holistic: orange | structured: cyan | per-image official raster IoU@0.50",
        fill=(65, 70, 80),
        font=_font(13),
    )
    y = header_height
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create auditable paired CULane qualitative figures.")
    parser.add_argument("--data-root", default="dataset")
    parser.add_argument("--holistic-pred-dir", required=True)
    parser.add_argument("--structured-pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gain-categories", nargs="+", default=["crowd", "shadow", "curve", "night"])
    parser.add_argument("--failure-categories", nargs="+", default=["curve", "cross"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--panel-width", type=int, default=600)
    args = parser.parse_args()

    categories = list(dict.fromkeys(args.gain_categories + args.failure_categories))
    unknown = sorted(set(categories) - set(CATEGORY_LISTS))
    if unknown:
        raise ValueError(f"Unknown categories: {unknown}")
    if args.workers < 1 or args.panel_width < 200:
        raise ValueError("workers must be positive and panel-width must be at least 200")

    tasks: list[tuple[str, str]] = []
    for category in categories:
        list_path = Path(CATEGORY_LISTS[category])
        if not list_path.is_absolute():
            list_path = Path.cwd() / list_path
        tasks.extend((category, rel) for rel in list_image_rel_paths(list_path))

    worker = partial(
        _evaluate_one,
        data_root=str(Path(args.data_root).resolve()),
        holistic_pred_dir=str(Path(args.holistic_pred_dir).resolve()),
        structured_pred_dir=str(Path(args.structured_pred_dir).resolve()),
        width=args.width,
        iou_threshold=float(args.iou_threshold),
    )
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(worker, tasks, chunksize=16), total=len(tasks), desc="paired metrics"))

    by_category = {category: [result for result in results if result.category == category] for category in categories}
    representative = [
        ("median positive F1 gain", _choose_representative_gain(by_category[category]))
        for category in args.gain_categories
    ]
    largest = [("largest F1 gain", _choose_largest_gain(by_category[category])) for category in args.gain_categories]
    failures = [("lowest structured F1" if category != "cross" else "most structured FP", _choose_failure(by_category[category], category)) for category in args.failure_categories]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    holistic_pred_dir = Path(args.holistic_pred_dir)
    structured_pred_dir = Path(args.structured_pred_dir)
    _make_montage(
        representative,
        data_root=data_root,
        holistic_pred_dir=holistic_pred_dir,
        structured_pred_dir=structured_pred_dir,
        output_path=output_dir / "representative_positive_gains.png",
        title="Matched CULane comparison: representative positive gains",
        panel_width=args.panel_width,
    )
    _make_montage(
        largest,
        data_root=data_root,
        holistic_pred_dir=holistic_pred_dir,
        structured_pred_dir=structured_pred_dir,
        output_path=output_dir / "largest_gains_diagnostic.png",
        title="Matched CULane comparison: largest category gains (diagnostic)",
        panel_width=args.panel_width,
    )
    _make_montage(
        failures,
        data_root=data_root,
        holistic_pred_dir=holistic_pred_dir,
        structured_pred_dir=structured_pred_dir,
        output_path=output_dir / "failure_cases.png",
        title="Structured-model failure cases",
        panel_width=args.panel_width,
    )

    with (output_dir / "selection_manifest.tsv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["figure", "selection_rule", *asdict(results[0]).keys(), "delta_f1"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for figure, selections in (
            ("representative_positive_gains.png", representative),
            ("largest_gains_diagnostic.png", largest),
            ("failure_cases.png", failures),
        ):
            for rule, result in selections:
                writer.writerow(
                    {
                        "figure": figure,
                        "selection_rule": rule,
                        **asdict(result),
                        "delta_f1": result.delta_f1,
                    }
                )


if __name__ == "__main__":
    main()
