from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


@dataclass
class TransformConfig:
    input_w: int = 800
    input_h: int = 288
    cut_height: int = 0
    horizontal_flip_prob: float = 0.0
    color_jitter: bool = False
    channel_shuffle_prob: float = 0.0
    hue_saturation_prob: float = 0.0
    blur_prob: float = 0.0
    affine_prob: float = 0.0
    affine_translate_x: float = 0.0
    affine_translate_y: float = 0.0
    affine_rotate_deg: float = 0.0
    affine_scale_min: float = 1.0
    affine_scale_max: float = 1.0
    random_shadow_prob: float = 0.0
    random_shadow_min_opacity: float = 0.25
    random_shadow_max_opacity: float = 0.55
    random_shadow_min_vertices: int = 3
    random_shadow_max_vertices: int = 6
    random_shadow_roi_start_y: float = 0.25


class LaneTransforms:
    def __init__(self, cfg: TransformConfig):
        self.cfg = cfg

    def __call__(
        self,
        image: Image.Image,
        lanes: list[list[tuple[float, float]]],
        seg_mask: Image.Image | None = None,
        training: bool = False,
    ) -> tuple[torch.Tensor, list[list[tuple[float, float]]], torch.Tensor | None, dict[str, float | bool]]:
        meta: dict[str, float | bool] = {"flipped": False, "crop_x": 0.0, "crop_y": 0.0}
        orig_w, orig_h = image.size
        crop_y = max(0, min(int(self.cfg.cut_height), orig_h - 1))
        if crop_y > 0:
            image = image.crop((0, crop_y, orig_w, orig_h))
            if seg_mask is not None:
                seg_mask = seg_mask.crop((0, crop_y, orig_w, orig_h))
            lanes = [[(x, y - crop_y) for x, y in lane] for lane in lanes]
            meta["crop_y"] = float(crop_y)
        crop_w, crop_h = image.size
        meta["crop_w"] = float(crop_w)
        meta["crop_h"] = float(crop_h)

        if training and self.cfg.affine_prob > 0 and np.random.random() < self.cfg.affine_prob:
            image, seg_mask, lanes = self._apply_affine(image, seg_mask, lanes)

        if training and self.cfg.horizontal_flip_prob > 0:
            if np.random.random() < self.cfg.horizontal_flip_prob:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if seg_mask is not None:
                    seg_mask = seg_mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                lanes = [[(crop_w - 1 - x, y) for x, y in lane] for lane in lanes]
                meta["flipped"] = True

        if training and self.cfg.color_jitter:
            factor = float(np.random.uniform(0.85, 1.15))
            image = ImageEnhance.Brightness(image).enhance(factor)
            factor = float(np.random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(factor)
        if training and self.cfg.hue_saturation_prob > 0 and np.random.random() < self.cfg.hue_saturation_prob:
            image = self._jitter_hue_saturation(image)
        if training and self.cfg.blur_prob > 0 and np.random.random() < self.cfg.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=float(np.random.uniform(0.4, 1.2))))
        if training and self.cfg.channel_shuffle_prob > 0 and np.random.random() < self.cfg.channel_shuffle_prob:
            arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
            order = np.random.permutation(3)
            image = Image.fromarray(arr[..., order], mode="RGB")
        if training and self.cfg.random_shadow_prob > 0 and np.random.random() < self.cfg.random_shadow_prob:
            image = self._apply_random_shadow(image)

        image = image.convert("RGB").resize(
            (self.cfg.input_w, self.cfg.input_h),
            resample=Image.Resampling.BILINEAR,
        )
        seg_tensor = None
        if seg_mask is not None:
            seg_mask = seg_mask.resize((self.cfg.input_w, self.cfg.input_h), resample=Image.Resampling.NEAREST)
            seg_arr = np.asarray(seg_mask, dtype=np.int64)
            if seg_arr.ndim == 3:
                seg_arr = seg_arr[..., 0]
            seg_tensor = torch.from_numpy(((seg_arr > 0) & (seg_arr < 255)).astype(np.float32))[None, ...]
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        tensor = (tensor - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        return tensor, lanes, seg_tensor, meta

    def _apply_affine(
        self,
        image: Image.Image,
        seg_mask: Image.Image | None,
        lanes: list[list[tuple[float, float]]],
    ) -> tuple[Image.Image, Image.Image | None, list[list[tuple[float, float]]]]:
        w, h = image.size
        angle = float(np.random.uniform(-self.cfg.affine_rotate_deg, self.cfg.affine_rotate_deg))
        scale = float(np.random.uniform(self.cfg.affine_scale_min, self.cfg.affine_scale_max))
        tx = float(np.random.uniform(-self.cfg.affine_translate_x, self.cfg.affine_translate_x) * w)
        ty = float(np.random.uniform(-self.cfg.affine_translate_y, self.cfg.affine_translate_y) * h)
        theta = np.deg2rad(angle)
        c = float(np.cos(theta) * scale)
        s = float(np.sin(theta) * scale)
        cx = (w - 1) * 0.5
        cy = (h - 1) * 0.5
        forward = np.array(
            [
                [c, -s, cx + tx - c * cx + s * cy],
                [s, c, cy + ty - s * cx - c * cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        inv = np.linalg.inv(forward)
        affine_data = tuple(inv[:2].reshape(-1).tolist())
        image = image.transform(
            (w, h),
            Image.Transform.AFFINE,
            affine_data,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        if seg_mask is not None:
            seg_mask = seg_mask.transform(
                (w, h),
                Image.Transform.AFFINE,
                affine_data,
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )
        out_lanes = []
        for lane in lanes:
            pts = []
            for x, y in lane:
                xp = forward[0, 0] * x + forward[0, 1] * y + forward[0, 2]
                yp = forward[1, 0] * x + forward[1, 1] * y + forward[1, 2]
                pts.append((float(xp), float(yp)))
            out_lanes.append(pts)
        return image, seg_mask, out_lanes

    @staticmethod
    def _jitter_hue_saturation(image: Image.Image) -> Image.Image:
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
        hue_delta = int(np.random.randint(-10, 11))
        sat_scale = float(np.random.uniform(0.85, 1.15))
        hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_delta) % 256).astype(np.uint8)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * sat_scale, 0, 255).astype(np.uint8)
        return Image.fromarray(hsv, mode="HSV").convert("RGB")

    def _apply_random_shadow(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        w, h = image.size
        roi_start = int(np.clip(self.cfg.random_shadow_roi_start_y, 0.0, 1.0) * h)
        min_vertices = max(3, int(self.cfg.random_shadow_min_vertices))
        max_vertices = max(min_vertices, int(self.cfg.random_shadow_max_vertices))
        num_vertices = int(np.random.randint(min_vertices, max_vertices + 1))
        points = []
        for _ in range(num_vertices):
            x = int(np.random.randint(0, max(w, 1)))
            y = int(np.random.randint(max(0, min(roi_start, h - 1)), max(h, 1)))
            points.append((x, y))
        min_opacity = max(0.0, self.cfg.random_shadow_min_opacity)
        max_opacity = min(1.0, self.cfg.random_shadow_max_opacity)
        if max_opacity < min_opacity:
            max_opacity = min_opacity
        opacity = float(np.random.uniform(min_opacity, max_opacity))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.polygon(points, fill=(0, 0, 0, int(255 * opacity)))
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
