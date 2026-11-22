"""Pilot run of the classical edits on a small sample.

Edits N source images with their recipes, then measures how much each edit
changed the image (LPIPS and SSIM against the unedited source). Both images
go through the full shared pipeline first:

    to_working_size() -> edit (classical only) -> standardize()

so the numbers reflect what the detectors will actually see.

Usage (from the repo root):
    python scripts/pilot_classical.py               # 50 images
    python scripts/pilot_classical.py --num 100

Outputs in data/scratch/pilot_classical/:
    pilot_metrics.csv     one row per image: recipe, strength, LPIPS, SSIM
    lpips_hist.png        distribution of LPIPS (compare with B's light-AI pilot)
    pairs/                original/edited pairs for the strongest and weakest edits
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidetect.data.classical import (  # noqa: E402
    aligned_reference, apply_recipe, make_recipe, metadata_row)
from aidetect.data.preprocess import standardize, to_working_size  # noqa: E402

OUT = ROOT / "data" / "scratch" / "pilot_classical"

# Rough bands for a "small" edit, to flag images worth a look.
# They are starting points: adjust after looking at the flagged pairs.
NO_OP_LPIPS = 0.01      # below this the edit is barely visible
TOO_STRONG_LPIPS = 0.30 # above this it's more than a touch-up


def load_lpips():
    import lpips
    import torch
    model = lpips.LPIPS(net="alex", verbose=False).eval()

    def score(a: Image.Image, b: Image.Image) -> float:
        def to_t(im):  
            x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)
            return x.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            return float(model(to_t(a), to_t(b)).item())
    return score


def ssim(a: Image.Image, b: Image.Image) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(np.asarray(a), np.asarray(b),
                                       channel_axis=2, data_range=255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
    pp = cfg["preprocess"]
    working, size, q = pp.get("working_size", 512), pp["size"], pp["jpeg_quality"]

    manifest = pd.read_csv(ROOT / "data" / "manifest_real.csv")
    manifest = manifest[manifest[["width", "height"]].min(axis=1) >= 256]  # skip tiny images
    sample = manifest.iloc[np.linspace(0, len(manifest) - 1, args.num).astype(int)]

    lpips_score = load_lpips()
    (OUT / "pairs").mkdir(parents=True, exist_ok=True)
    rows, images = [], {}
    for _, r in sample.iterrows():
        base = to_working_size(Image.open(ROOT / "data" / r["path"]), working)
        recipe = make_recipe(r["source_id"])
        
        real_std = standardize(aligned_reference(base, recipe), size, q)
        edit_std = standardize(apply_recipe(base, recipe), size, q)

        row = metadata_row(recipe, path="", split=r["split"])
        row["ops"] = " + ".join(op["name"] for op in recipe["ops"])
        row["lpips"] = round(lpips_score(real_std, edit_std), 4)
        row["ssim"] = round(ssim(real_std, edit_std), 4)
        rows.append(row)
        images[r["source_id"]] = (real_std, edit_std)

    df = pd.DataFrame(rows)
    df.drop(columns=["path", "mask_area", "class", "editor"]).to_csv(
        OUT / "pilot_metrics.csv", index=False)

    # Summary per main op
    print(f"\nPilot on {len(df)} images\n")
    summary = df.groupby("edit_type")[["lpips", "ssim"]].agg(["mean", "min", "max"])
    print(summary.round(3).to_string())
    print(f"\nOverall LPIPS: median {df.lpips.median():.3f}, "
          f"range {df.lpips.min():.3f}-{df.lpips.max():.3f}")

    # A crop-only recipe scores ~0 against its aligned reference by design
    # (framing changed, content didn't), so don't flag those as no-ops.
    crop_only = df.ops == "crop"
    flagged = df[((df.lpips < NO_OP_LPIPS) & ~crop_only) | (df.lpips > TOO_STRONG_LPIPS)]
    print(f"\nFlagged {len(flagged)} image(s): LPIPS < {NO_OP_LPIPS} (barely visible) "
          f"or > {TOO_STRONG_LPIPS} (too strong)")
    for _, f in flagged.iterrows():
        print(f"  {f.source_id}  lpips={f.lpips:.3f}  ops: {f.ops}")

    # Side-by-side pairs: the 5 weakest, the 5 strongest, and every flagged one
    picks = set(df.nsmallest(5, "lpips").source_id) | set(df.nlargest(5, "lpips").source_id) \
        | set(flagged.source_id)
    for sid in picks:
        a, b = images[sid]
        pair = Image.new("RGB", (a.width * 2 + 8, a.height), "white")
        pair.paste(a, (0, 0))
        pair.paste(b, (a.width + 8, 0))
        lp = df.loc[df.source_id == sid, "lpips"].item()
        pair.save(OUT / "pairs" / f"lpips{lp:.3f}_{sid}.png")

    # LPIPS histogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df.lpips, bins=20, color="#4C72B0")
    ax.set_xlabel("LPIPS vs. unedited source (higher = more change)")
    ax.set_ylabel("Images")
    ax.set_title(f"Classical edits, pilot (n={len(df)})")
    fig.tight_layout()
    fig.savefig(OUT / "lpips_hist.png", dpi=150)

    print(f"\nSaved metrics, histogram and pairs to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()