"""Classical half of the matched-pair set (evaluation only).

For every source image, makes each lighting edit that InstructPix2Pix makes in
the matched set (configs/diffusion.yaml: edits.ip2p.instructions) with the
classical tools instead: one single operation, no extras, with the direction
fixed and the size drawn from the classical pipeline's own ranges.

A classical image and the IP2P image with the same (source_id, edit_type,
direction) then differ in the tool only, which is the "tool vs. content" test.

Same pipeline as every other class:
    to_working_size() -> edit -> save_standardized() (one JPEG encode)

Unlike make_classical.py there is no LPIPS filter: no source is dropped, so
every IP2P image has its partner.

Usage (CPU; from the repo root):
    python scripts/make_matched_classical.py              # every usable source
    python scripts/make_matched_classical.py --num 200    # the small batch

Outputs (paths relative to data/):
    processed/matched/<source_id>__classical_<edit_type>_<direction>.jpg
    processed/metadata_matched_classical.csv    contract columns + direction
"""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.classical import RANGES, apply_op, strength_of  # noqa: E402
from src.data.contract import METADATA_COLUMNS, assign_split  # noqa: E402
from src.data.preprocess import save_standardized, to_working_size  # noqa: E402

DATA_CFG = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
DIFF_CFG = yaml.safe_load(open(ROOT / "configs" / "diffusion.yaml"))
DATA = ROOT / DATA_CFG["data_root"]
PP, SPLITS = DATA_CFG["preprocess"], DATA_CFG["splits"]
EDITS = sorted({(i["edit_type"], i["direction"])
                for i in DIFF_CFG["edits"]["ip2p"]["instructions"]})
OUT_DIR = Path(DIFF_CFG["matched"]["out_dir"])
META = DATA / "processed" / "metadata_matched_classical.csv"


def sample_params(edit_type: str, direction: str, rng: np.random.Generator) -> dict:
    """Parameters for one op with the direction fixed, from the classical ranges."""
    r = RANGES[edit_type]
    if edit_type == "brightness":
        lo, hi = (1.0 + r["gap"], r["hi"]) if direction == "up" else (r["lo"], 1.0 - r["gap"])
        return {"factor": float(rng.uniform(lo, hi))}
    if edit_type == "white_balance":
        mag = float(rng.uniform(r["min_abs"], r["max_abs"]))
        return {"shift": mag if direction == "up" else -mag}
    raise KeyError(f"No matched classical edit for {edit_type!r}")


def process(s: dict) -> list[dict]:
    base = to_working_size(Image.open(DATA / s["path"]), PP.get("working_size", 512))
    rows = []
    for edit_type, direction in EDITS:
        image_id = f"{s['source_id']}__classical_{edit_type}_{direction}"
        seed = int(hashlib.sha256(f"matched:{image_id}".encode()).hexdigest()[:8], 16)
        params = sample_params(edit_type, direction, np.random.default_rng(seed))
        rel = OUT_DIR / f"{image_id}.jpg"
        save_standardized(apply_op(base, edit_type, params), DATA / rel,
                          PP["size"], PP["jpeg_quality"])
        rows.append({
            "image_id": image_id,
            "source_id": s["source_id"],
            "path": str(rel),
            "class": "classical",
            "edit_type": edit_type,
            "strength": round(strength_of(edit_type, params), 4),
            "mask_area": "",
            "editor": "pil",
            "split": assign_split(s["source_id"], SPLITS["train"], SPLITS["val"]),
            "params": json.dumps([{"name": edit_type, "params": params}]),
            "direction": direction,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, help="only the first N usable sources")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    min_side = PP.get("min_short_side", 256)
    with open(DATA / "manifest_real.csv") as f:
        sources = [r for r in csv.DictReader(f)
                   if min(int(r["width"]), int(r["height"])) >= min_side][:args.num]
    with mp.Pool(args.workers) as pool:
        rows = [row for rows in tqdm(pool.imap(process, sources, chunksize=8),
                                     total=len(sources), desc="matched classical")
                for row in rows]
    with open(META, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_COLUMNS + ["direction"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {META.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
