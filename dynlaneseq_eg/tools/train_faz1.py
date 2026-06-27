"""Faz 1 eğitim aracı: S0 frozen, sadece quality head eğitilir.

Mevcut train.py ile aynı altyapıyı kullanır. Fark:
  - --init-from ile S0 checkpoint yüklenir
  - Tüm parametreler dondurulur (requires_grad=False)
  - SADECE structured_query_head.quality parametreleri açılır
  - Bu sayede optimizer sıfır param grubu hatası vermez

Kullanım:
    python -m dynlaneseq_eg.tools.train_faz1 \\
        --config dynlaneseq_eg/configs/culane_faz1_soft_quality.yaml \\
        --init-from outputs/culane_s0_structured_query_res34_b16_50ep/iter_0175000.pt \\
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_compatible_model_weights, save_checkpoint
from dynlaneseq_eg.engine.logger import SmoothedLogger
from dynlaneseq_eg.engine.train_one_epoch import train_one_epoch
from dynlaneseq_eg.engine.visualizer import save_prediction_visuals
from dynlaneseq_eg.factory import build_criterion, build_dataloader, build_matcher, build_model, build_optimizer, build_scheduler


def freeze_all_except_quality(model: torch.nn.Module) -> dict[str, int]:
    """S0 modelini dondur, sadece quality head'i eğitilebilir bırak.

    Returns:
        dict: frozen_params, trainable_params sayıları
    """
    frozen = 0
    trainable = 0

    # Önce hepsini dondur
    for param in model.parameters():
        param.requires_grad_(False)
        frozen += param.numel()

    # Sadece quality head'i aç
    # DynLaneSeqS0.structured_query_head.quality = nn.Sequential(Linear, GELU, Linear)
    quality_opened = 0
    for name, param in model.named_parameters():
        if "structured_query_head.quality" in name:
            param.requires_grad_(True)
            frozen -= param.numel()
            trainable += param.numel()
            quality_opened += param.numel()
            print(f"  [TRAINABLE] {name}  ({param.numel():,} params)")

    if quality_opened == 0:
        raise RuntimeError(
            "structured_query_head.quality parametreleri bulunamadı! "
            "Model yapısı değişmiş olabilir. Model parametrelerini kontrol edin."
        )

    return {"frozen": frozen, "trainable": trainable}


def main() -> None:
    parser = argparse.ArgumentParser(description="Faz 1: S0 frozen, quality head only training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init-from", required=True, help="S0 checkpoint yolu")
    parser.add_argument("--max-iters", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("training", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device)
    train_cfg = cfg.get("training", {})

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", False))
        if bool(train_cfg.get("tf32", False)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    # Model yükle
    model = build_model(cfg).to(device)

    # S0 ağırlıklarını yükle
    print(f"\nS0 checkpoint yükleniyor: {args.init_from}")
    stats = load_compatible_model_weights(args.init_from, model)
    print(f"Yükleme istatistikleri: {stats}")

    # Freeze: sadece quality head açık
    print("\nParametreler dondurluyor (quality head hariç)...")
    freeze_stats = freeze_all_except_quality(model)
    print(f"\nFreeze özeti:")
    print(f"  Frozen    : {freeze_stats['frozen']:>12,} parametre")
    print(f"  Trainable : {freeze_stats['trainable']:>12,} parametre  ← sadece quality head")

    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    # build_optimizer sadece requires_grad=True olanları alır
    optimizer = build_optimizer(cfg, model)

    # Optimizer gruplarını kontrol et
    total_opt_params = sum(len(g["params"]) for g in optimizer.param_groups)
    print(f"\nOptimizer: {total_opt_params} parametre grubu aktif")
    for g in optimizer.param_groups:
        if len(g["params"]) > 0:
            print(f"  [{g.get('name', '?')}] lr={g['lr']:.2e}, n_params={len(g['params'])}")

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(cfg.get("training", {}).get("amp", False) and device.type == "cuda")
    )
    loader = build_dataloader(cfg, split="train", training=True)
    out_dir = Path(cfg.get("output_dir", "outputs/faz1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    vis_interval = int(cfg.get("training", {}).get("vis_interval", 2000))
    planned_iters = args.max_iters or int(cfg.get("training", {}).get("max_iters", len(loader)))
    scheduler = build_scheduler(cfg, optimizer, total_iters=planned_iters)

    print(f"\n{'='*60}")
    print(f"FAZ 1 EĞİTİMİ BAŞLIYOR")
    print(f"  Config       : {args.config}")
    print(f"  Checkpoint   : {args.init_from}")
    print(f"  Output       : {out_dir}")
    print(f"  Iter         : {planned_iters:,}")
    print(f"  Batch size   : {train_cfg.get('batch_size', 16)}")
    print(f"  Device       : {device}")
    print(f"  AMP          : {bool(train_cfg.get('amp', False))}")
    print(f"  Trainable    : {freeze_stats['trainable']:,} params (quality head only)")
    print(f"{'='*60}\n")

    def vis(images, targets, metas, outputs, iteration):
        if iteration % vis_interval == 0:
            save_prediction_visuals(images, targets, metas, outputs, out_dir / "vis", iteration)

    checkpoint_interval = int(cfg.get("training", {}).get("checkpoint_interval", 10000))

    def save_periodic(iteration: int):
        if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            save_checkpoint(
                out_dir / f"iter_{iteration:07d}.pt",
                model, optimizer, scaler, iteration, cfg, scheduler=scheduler
            )

    end_iter = train_one_epoch(
        model,
        loader,
        matcher,
        criterion,
        optimizer,
        device,
        cfg,
        start_iter=0,
        max_iters=planned_iters,
        scaler=scaler,
        scheduler=scheduler,
        logger=SmoothedLogger(),
        visualizer=vis,
        checkpoint_saver=save_periodic,
    )

    save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, end_iter, cfg, scheduler=scheduler)
    save_checkpoint(out_dir / f"iter_{end_iter:07d}.pt", model, optimizer, scaler, end_iter, cfg, scheduler=scheduler)
    print(f"\nFaz 1 eğitimi tamamlandı. Checkpoint: {out_dir}/last.pt")


if __name__ == "__main__":
    main()
