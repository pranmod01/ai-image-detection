"""Step 8: generate the full classical-edit set (and the matching real set).

For every source image in the manifest (skipping tiny ones):

    to_working_size() -> recipe edit -> standardize() -> save as JPEG

Each edit is checked against the unedited source (LPIPS, with the same crop
applied to the reference; see `aligned_reference`). If it changed the image
too little (< --floor) or too much (> --ceiling), its parameters are redrawn,
up to --max-attempts tries in total. Sources that never pass are dropped and
listed in dropped_classical.csv.

`--real` writes the real class through the exact same pipeline (no edit), so
real and classical images are resized and JPEG-encoded identically.

Usage (from the repo root):
    python scripts/make_classical.py --num 20          # quick test
    python scripts/make_classical.py                   # full classical set
    python scripts/make_classical.py --real            # full real set

Safe to stop and rerun: sources already in the metadata CSV are skipped.

Outputs (paths relative to data/):
    processed/classical/<source_id>__classical.jpg
    processed/metadata_classical.csv     contract columns + params, lpips, ssim, attempt
    processed/dropped_classical.csv      sources that never passed the check
    processed/real/<source_id>.jpg       (with --real)
    processed/metadata_real.csv          (with --real)
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidetect.data.classical import (  # noqa: E402
    aligned_reference, apply_recipe, make_recipe, metadata_row)
from aidetect.data.contract import METADATA_COLUMNS  # noqa: E402
from aidetect.data.preprocess import (  # noqa: E402
    resize_center_crop, standardize, to_working_size)

DATA = ROOT / "data"
OUT = DATA / "processed"

CFG = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
PP = CFG["preprocess"]
WORKING = PP.get("working_size", 512)
SIZE, QUALITY = PP["size"], PP["jpeg_quality"]
MIN_SHORT_SIDE = PP.get("min_short_side", 256)

CLASSICAL_COLS = METADATA_COLUMNS + ["params", "lpips", "ssim", "attempt"]
DROPPED_COLS = ["source_id", "split", "best_lpips", "attempts"]


# ---------------------------------------------------------------------------
# Saving: one JPEG encode at the shared quality, identical to standardize()
# ---------------------------------------------------------------------------
def save_standardized(img: Image.Image, path: Path) -> None:
    """Resize + crop, then a single JPEG save at the shared quality.

    Reading the file back gives exactly what standardize() returns, so the
    saved file is what every detector sees (no second JPEG encode).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    resize_center_crop(img, SIZE).save(path, format="JPEG", quality=QUALITY)


# ---------------------------------------------------------------------------
# Metrics (one LPIPS model per worker process)
# ---------------------------------------------------------------------------
_lpips_model = None


def _init_worker():
    global _lpips_model
    import lpips
    import torch
    torch.set_num_threads(1)     # one core per worker; the pool provides parallelism
    _lpips_model = lpips.LPIPS(net="alex", verbose=False).eval()


def lpips_score(a: Image.Image, b: Image.Image) -> float:
    import torch

    def to_t(im):
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)
        return x.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        return float(_lpips_model(to_t(a), to_t(b)).item())


def ssim_score(a: Image.Image, b: Image.Image) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(np.asarray(a), np.asarray(b),
                                       channel_axis=2, data_range=255))


# ---------------------------------------------------------------------------
# Per-source work
# ---------------------------------------------------------------------------
def process_classical(task):
    r, floor, ceiling, max_attempts = task
    base = to_working_size(Image.open(DATA / r["path"]), WORKING)
    best_lp = None                     # for the dropped log only
    for attempt in range(max_attempts):
        recipe = make_recipe(r["source_id"], attempt=attempt)
        edited = apply_recipe(base, recipe)
        ref_std = standardize(aligned_reference(base, recipe), SIZE, QUALITY)
        edit_std = standardize(edited, SIZE, QUALITY)
        lp = lpips_score(ref_std, edit_std)
        if best_lp is None or abs(lp - floor) < abs(best_lp - floor):
            best_lp = lp
        if floor <= lp <= ceiling:
            rel = Path("processed") / "classical" / f"{r['source_id']}__classical.jpg"
            save_standardized(edited, DATA / rel)
            row = metadata_row(recipe, path=str(rel), split=r["split"])
            row.update(lpips=round(lp, 4), ssim=round(ssim_score(ref_std, edit_std), 4),
                       attempt=attempt)
            return "kept", row
    return "dropped", {"source_id": r["source_id"], "split": r["split"],
                       "best_lpips": round(best_lp, 4), "attempts": max_attempts}


def process_real(r):
    base = to_working_size(Image.open(DATA / r["path"]), WORKING)
    rel = Path("processed") / "real" / f"{r['source_id']}.jpg"
    save_standardized(base, DATA / rel)
    return "kept", {"image_id": r["source_id"], "source_id": r["source_id"],
                    "path": str(rel), "class": "real", "edit_type": "none",
                    "strength": 0, "mask_area": "", "editor": "none",
                    "split": r["split"]}


# ---------------------------------------------------------------------------
# CSV helpers (append, so a stopped run can resume)
# ---------------------------------------------------------------------------
def done_ids(*paths: Path) -> set:
    ids = set()
    for p in paths:
        if p.exists():
            ids |= set(pd.read_csv(p, usecols=["source_id"]).source_id)
    return ids


def appender(path: Path, cols: list):
    new = not path.exists()
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    if new:
        w.writeheader()
    return f, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="write the real class instead")
    ap.add_argument("--num", type=int, help="only the first N sources (quick test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--floor", type=float, default=0.015, help="min LPIPS to keep an edit")
    ap.add_argument("--ceiling", type=float, default=0.30, help="max LPIPS to keep an edit")
    ap.add_argument("--max-attempts", type=int, default=4, help="tries per source")
    args = ap.parse_args()

    manifest = pd.read_csv(DATA / "manifest_real.csv")
    manifest = manifest[manifest[["width", "height"]].min(axis=1) >= MIN_SHORT_SIDE]
    if args.num:
        manifest = manifest.head(args.num)

    OUT.mkdir(parents=True, exist_ok=True)
    if args.real:
        meta = OUT / "metadata_real.csv"
        todo_skip = done_ids(meta)
        fn, cols, dropped_path = process_real, METADATA_COLUMNS, None
    else:
        meta = OUT / "metadata_classical.csv"
        dropped_path = OUT / "dropped_classical.csv"
        todo_skip = done_ids(meta, dropped_path)
        fn, cols = process_classical, CLASSICAL_COLS

    rows = [r for r in manifest.to_dict("records") if r["source_id"] not in todo_skip]
    print(f"{len(manifest)} sources (short side >= {MIN_SHORT_SIDE}px), "
          f"{len(todo_skip)} already done, {len(rows)} to process")
    if not rows:
        return
    tasks = rows if args.real else [(r, args.floor, args.ceiling, args.max_attempts)
                                    for r in rows]

    meta_f, meta_w = appender(meta, cols)
    drop_f, drop_w = appender(dropped_path, DROPPED_COLS) if dropped_path else (None, None)
    counts = {"kept": 0, "dropped": 0}
    try:
        if args.workers == 1:
            if not args.real:
                _init_worker()
            results = map(fn, tasks)
        else:
            pool = mp.Pool(args.workers, initializer=None if args.real else _init_worker)
            results = pool.imap_unordered(fn, tasks, chunksize=8)
        for status, row in tqdm(results, total=len(tasks), desc="classical" if not args.real else "real"):
            counts[status] += 1
            (meta_w if status == "kept" else drop_w).writerow(row)
            if sum(counts.values()) % 50 == 0:   # flush regularly so a crash loses little
                meta_f.flush()
                if drop_f:
                    drop_f.flush()
    finally:
        meta_f.close()
        if drop_f:
            drop_f.close()

    print(f"Done: kept {counts['kept']}, dropped {counts['dropped']}.")
    print(f"Metadata: {meta.relative_to(ROOT)}")
    if not args.real and counts["kept"]:
        df = pd.read_csv(meta)
        print(f"LPIPS median {df.lpips.median():.3f}; attempts needed: "
              f"{df.attempt.value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()