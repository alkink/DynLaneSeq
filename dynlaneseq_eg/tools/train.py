from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dynlaneseq_eg.config import load_config
from dynlaneseq_eg.engine.checkpoint import load_checkpoint, load_compatible_model_weights, save_checkpoint
from dynlaneseq_eg.engine.logger import SmoothedLogger
from dynlaneseq_eg.engine.train_one_epoch import train_one_epoch
from dynlaneseq_eg.engine.visualizer import save_prediction_visuals
from dynlaneseq_eg.factory import build_criterion, build_dataloader, build_matcher, build_model, build_optimizer, build_scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-from", default="", help="Initialize compatible model weights only; optimizer and iteration stay fresh.")
    parser.add_argument("--max-iters", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
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
    model = build_model(cfg).to(device)
    matcher = build_matcher(cfg)
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("training", {}).get("amp", False) and device.type == "cuda"))
    loader = build_dataloader(cfg, split="train", training=True)
    out_dir = Path(cfg.get("output_dir", "outputs/train"))
    vis_interval = int(cfg.get("training", {}).get("vis_interval", 100))
    planned_iters = args.max_iters or int(cfg.get("training", {}).get("max_iters", len(loader)))
    scheduler = build_scheduler(cfg, optimizer, total_iters=planned_iters)
    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    start_iter = 0
    if args.init_from:
        stats = load_compatible_model_weights(args.init_from, model)
        print(f"initialized compatible weights from {args.init_from}: {stats}")
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer, scaler, strict=False, scheduler=scheduler)
    batch_size = int(cfg.get("training", {}).get("batch_size", 1))
    approx_epochs = planned_iters / max(len(loader), 1)
    print(
        {
            "model": cfg.get("model", {}).get("name", "DynLaneSeq"),
            "output_dir": str(out_dir),
            "device": str(device),
            "train_images": len(loader.dataset),
            "batch_size": batch_size,
            "iters": planned_iters,
            "start_iter": start_iter,
            "approx_epochs_this_run": round(approx_epochs, 2),
            "amp": bool(cfg.get("training", {}).get("amp", False) and device.type == "cuda"),
            "log_interval": int(cfg.get("training", {}).get("log_interval", 10)),
            "scheduler": cfg.get("scheduler", {"name": "none"}),
        }
    )

    def vis(images, targets, metas, outputs, iteration):
        if iteration % vis_interval == 0:
            save_prediction_visuals(images, targets, metas, outputs, out_dir / "vis", iteration)

    checkpoint_interval = int(cfg.get("training", {}).get("checkpoint_interval", 0))

    def save_periodic(iteration: int):
        if checkpoint_interval > 0 and iteration % checkpoint_interval == 0:
            save_checkpoint(out_dir / f"iter_{iteration:07d}.pt", model, optimizer, scaler, iteration, cfg, scheduler=scheduler)

    end_iter = train_one_epoch(
        model,
        loader,
        matcher,
        criterion,
        optimizer,
        device,
        cfg,
        start_iter=start_iter,
        max_iters=planned_iters,
        scaler=scaler,
        scheduler=scheduler,
        logger=SmoothedLogger(),
        visualizer=vis,
        checkpoint_saver=save_periodic,
    )
    save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, end_iter, cfg, scheduler=scheduler)
    save_checkpoint(out_dir / f"iter_{end_iter:07d}.pt", model, optimizer, scaler, end_iter, cfg, scheduler=scheduler)


if __name__ == "__main__":
    main()
