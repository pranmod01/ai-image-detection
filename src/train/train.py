"""Shared training script. Every detector is trained and evaluated through this
file, so labels, losses and saved predictions are the same for all models.

  python -m src.train.train --config configs/train.yaml

Label setups:
  ordinal     K-1 sigmoid outputs, output k predicts "class index > k"
              (cumulative-link / CORAL-style); prediction = number of outputs > 0.5.
  regression  one output, MSE on the edit-magnitude target (see add_targets).

Writes to <out_root>/<run_name>/: config.yaml (resolved), metrics.jsonl (one line
per epoch), best.pt (lowest val loss), predictions_{val,test}.csv. The
prediction CSVs are the input to the shared evaluation code.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.contract import CLASSES
from src.data.dataset import EditDataset, add_targets, load_metadata
from src.models import build_model

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def num_outputs(label_setup: str) -> int:
    return len(CLASSES) - 1 if label_setup == "ordinal" else 1


def loss_fn(out: torch.Tensor, y: torch.Tensor, label_setup: str) -> torch.Tensor:
    if label_setup == "ordinal":
        k = torch.arange(out.shape[1], device=out.device)
        return F.binary_cross_entropy_with_logits(out, (y[:, None] > k).float())
    return F.mse_loss(out[:, 0], y)


def predict(out: torch.Tensor, label_setup: str) -> torch.Tensor:
    if label_setup == "ordinal":
        return (out.sigmoid() > 0.5).sum(1).float()
    return out[:, 0]


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    """Returns (summary metrics, per-image predictions)."""
    model.eval()
    outs, ys, idx = [], [], []
    for x, y, i in loader:
        outs.append(model(x.to(device)).float().cpu())
        ys.append(y)
        idx.append(i)
    out, y, idx = torch.cat(outs), torch.cat(ys), torch.cat(idx)
    pred = predict(out, cfg["label_setup"])
    metrics = {"loss": loss_fn(out, y, cfg["label_setup"]).item(),
               "mae": (pred - y).abs().mean().item()}
    if cfg["label_setup"] == "ordinal":
        metrics["acc"] = (pred == y).float().mean().item()
    preds = pd.DataFrame({"row": idx.numpy(), "target": y.numpy(), "pred": pred.numpy()})
    for k in range(out.shape[1]):
        preds[f"out_{k}"] = out[:, k].numpy()
    return metrics, preds


def make_loaders(df, data_root, cfg):
    g = torch.Generator().manual_seed(cfg["seed"])
    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        if part.empty:
            raise ValueError(f"no images in split {split!r}; is the metadata too small?")
        ds = EditDataset(part, data_root)
        loaders[split] = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=split == "train",
                                    num_workers=cfg["num_workers"], generator=g,
                                    pin_memory=torch.cuda.is_available())
    return loaders


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg = yaml.safe_load(open(ROOT / cfg["data_config"]))
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = add_targets(load_metadata([ROOT / p for p in cfg["metadata"]]), cfg)
    loaders = make_loaders(df, ROOT / data_cfg["data_root"], cfg)
    counts = df.groupby(["split", "class"]).size().unstack(fill_value=0)
    print(f"{len(df)} images\n{counts}")

    run_dir = ROOT / cfg["out_root"] / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"train": cfg, "data": data_cfg}, open(run_dir / "config.yaml", "w"),
                   sort_keys=False)

    model = build_model(cfg["model"], num_outputs=num_outputs(cfg["label_setup"]),
                        **cfg.get("model_kwargs", {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best = float("inf")
    with open(run_dir / "metrics.jsonl", "w") as log:
        for epoch in range(cfg["epochs"]):
            model.train()
            total, n = 0.0, 0
            for x, y, _ in loaders["train"]:
                x, y = x.to(device), y.to(device)
                loss = loss_fn(model(x), y, cfg["label_setup"])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, n = total + loss.item() * len(y), n + len(y)
            val, _ = evaluate(model, loaders["val"], cfg, device)
            rec = {"epoch": epoch, "train_loss": total / n, **{f"val_{k}": v for k, v in val.items()}}
            print(json.dumps(rec))
            log.write(json.dumps(rec) + "\n")
            if val["loss"] < best:
                best = val["loss"]
                torch.save(model.state_dict(), run_dir / "best.pt")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    for split in ("val", "test"):
        metrics, preds = evaluate(model, loaders[split], cfg, device)
        meta = loaders[split].dataset.df.iloc[preds["row"]].reset_index(drop=True)
        preds = pd.concat([meta, preds.drop(columns=["row", "target"])], axis=1)
        preds.to_csv(run_dir / f"predictions_{split}.csv", index=False)
        print(f"best checkpoint, {split}: {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
