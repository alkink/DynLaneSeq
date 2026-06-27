from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from dynlaneseq_eg.evaluation.culane_metric import culane_metric, load_culane_img_data


CATEGORIES = {
    "normal": "test0_normal.txt",
    "crowd": "test1_crowd.txt",
    "hlight": "test2_hlight.txt",
    "shadow": "test3_shadow.txt",
    "noline": "test4_noline.txt",
    "arrow": "test5_arrow.txt",
    "curve": "test6_curve.txt",
    "cross": "test7_cross.txt",
    "night": "test8_night.txt",
}


@dataclass(frozen=True)
class FailureRecord:
    category: str
    rel_image: str
    tp: int
    fp: int
    fn: int
    pred_count: int
    gt_count: int

    @property
    def failed(self) -> bool:
        return self.fp > 0 or self.fn > 0

    @property
    def severity(self) -> tuple[int, int, int, int]:
        # Prefer visually informative misses: false negatives first, then false positives.
        return (self.fn + self.fp, self.fn, self.fp, self.gt_count)


def _read_rel_paths(list_path: Path) -> list[str]:
    rels: list[str] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel = line.split()[0]
            rels.append(rel[1:] if rel.startswith("/") else rel)
    return rels


def _metric_task(args: tuple[str, str, str, str, float, int]) -> FailureRecord:
    category, rel_image, dataset_root, pred_dir, iou_threshold, line_width = args
    dataset_root_p = Path(dataset_root)
    pred_dir_p = Path(pred_dir)
    anno_path = dataset_root_p / rel_image.replace(".jpg", ".lines.txt")
    pred_path = pred_dir_p / rel_image.replace(".jpg", ".lines.txt")
    gt = load_culane_img_data(anno_path)
    pred = load_culane_img_data(pred_path)
    metric = culane_metric(
        pred,
        gt,
        width=int(line_width),
        iou_thresholds=(float(iou_threshold),),
        official=True,
        img_shape=(590, 1640, 3),
    )[float(iou_threshold)]
    tp, fp, fn = (int(metric[0]), int(metric[1]), int(metric[2]))
    return FailureRecord(
        category=category,
        rel_image=rel_image,
        tp=tp,
        fp=fp,
        fn=fn,
        pred_count=len(pred),
        gt_count=len(gt),
    )


def _draw_polyline(img: np.ndarray, lane: list[tuple[float, float]], color: tuple[int, int, int], thickness: int) -> None:
    if len(lane) < 2:
        return
    pts = [(int(round(x)), int(round(y))) for x, y in lane]
    for p1, p2 in zip(pts[:-1], pts[1:]):
        cv2.line(img, p1, p2, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, max(2, thickness // 2), color, -1, lineType=cv2.LINE_AA)


def _put_label(img: np.ndarray, lines: Iterable[str]) -> None:
    lines = list(lines)
    if not lines:
        return
    pad = 8
    line_h = 24
    width = min(img.shape[1] - 2 * pad, max(520, max(len(line) for line in lines) * 11))
    height = pad * 2 + line_h * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    for idx, line in enumerate(lines):
        y = pad + 18 + idx * line_h
        cv2.putText(img, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def draw_overlay(record: FailureRecord, dataset_root: Path, pred_dir: Path, out_path: Path) -> bool:
    img_path = dataset_root / record.rel_image
    anno_path = dataset_root / record.rel_image.replace(".jpg", ".lines.txt")
    pred_path = pred_dir / record.rel_image.replace(".jpg", ".lines.txt")
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return False
    gt = load_culane_img_data(anno_path)
    pred = load_culane_img_data(pred_path)
    canvas = img.copy()
    # Draw GT first as thicker red, predictions second as thinner green.
    # BGR: red=(0,0,255), green=(0,255,0).
    for lane in gt:
        _draw_polyline(canvas, lane, (0, 0, 255), 6)
    for lane in pred:
        _draw_polyline(canvas, lane, (0, 255, 0), 3)
    _put_label(
        canvas,
        [
            f"{record.category} | TP={record.tp} FP={record.fp} FN={record.fn} | GT={record.gt_count} Pred={record.pred_count}",
            "GT: red | Pred: green",
            record.rel_image,
        ],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), canvas))


def make_contact_sheet(image_paths: list[Path], out_path: Path, thumb_w: int = 410, cols: int = 2) -> None:
    thumbs: list[np.ndarray] = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        scale = thumb_w / float(img.shape[1])
        thumb_h = max(1, int(round(img.shape[0] * scale)))
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
    if not thumbs:
        return
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.shape[0] for t in thumbs)
    sheet = np.full((rows * cell_h, cols * thumb_w, 3), 32, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        y0 = r * cell_h
        x0 = c * thumb_w
        sheet[y0 : y0 + thumb.shape[0], x0 : x0 + thumb.shape[1]] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw CULane failed test images with GT red and predictions green.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--split-dir", default="dataset/list/test_split")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-per-category", type=int, default=10)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--line-width", type=int, default=30)
    parser.add_argument("--workers", type=int, default=max(1, min(cpu_count(), 12)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    pred_dir = Path(args.pred_dir)
    split_dir = Path(args.split_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_lines = [
        "# CULane S0 Failure Overlays",
        "",
        f"Prediction dir: `{pred_dir}`",
        f"IoU threshold: `{args.iou_threshold}`",
        "",
        "GT is red. Prediction is green.",
        "",
    ]
    all_summary: list[dict[str, object]] = []
    for category, filename in CATEGORIES.items():
        list_path = split_dir / filename
        if not list_path.exists():
            print(f"missing category list: {list_path}")
            continue
        rels = _read_rel_paths(list_path)
        tasks = [
            (category, rel, str(dataset_root), str(pred_dir), float(args.iou_threshold), int(args.line_width))
            for rel in rels
        ]
        if args.workers > 1:
            with Pool(args.workers) as pool:
                records = list(
                    tqdm(
                        pool.imap_unordered(_metric_task, tasks, chunksize=32),
                        total=len(tasks),
                        desc=f"scoring {category}",
                        ncols=90,
                    )
                )
        else:
            records = [_metric_task(task) for task in tqdm(tasks, desc=f"scoring {category}", ncols=90)]
        failed = [record for record in records if record.failed]
        failed.sort(key=lambda record: record.severity, reverse=True)
        selected = failed[: int(args.num_per_category)]
        category_dir = out_dir / category
        written_paths: list[Path] = []
        for rank, record in enumerate(selected, start=1):
            safe_name = record.rel_image.replace("/", "__").replace(".jpg", "")
            out_path = category_dir / f"{rank:02d}_tp{record.tp}_fp{record.fp}_fn{record.fn}_{safe_name}.jpg"
            if draw_overlay(record, dataset_root, pred_dir, out_path):
                written_paths.append(out_path)
        sheet_path = out_dir / f"{category}_contact_sheet.jpg"
        make_contact_sheet(written_paths, sheet_path)
        summary = {
            "category": category,
            "images": len(records),
            "failed": len(failed),
            "selected": len(written_paths),
            "contact_sheet": str(sheet_path),
        }
        all_summary.append(summary)
        index_lines.extend(
            [
                f"## {category}",
                "",
                f"- Images: {len(records)}",
                f"- Failed images: {len(failed)}",
                f"- Contact sheet: [{sheet_path.name}]({sheet_path.name})",
                "",
            ]
        )
        for path in written_paths:
            rel_path = path.relative_to(out_dir)
            index_lines.append(f"- [{rel_path}]({rel_path})")
        index_lines.append("")
        print(f"{category}: failed={len(failed)} selected={len(written_paths)} out={category_dir}")

    (out_dir / "index.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"index: {out_dir / 'index.md'}")


if __name__ == "__main__":
    main()
