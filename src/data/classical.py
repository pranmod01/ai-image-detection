"""Classical (non-AI) photo edits: the "Photoshop" class.

Each operation has two parts:
  - apply_<op>(img, **params) -> Image   does the edit, deterministically
  - sample_<op>(rng) -> dict             draws realistic parameters

Edits run on the 512px working copy (`to_working_size`), and `standardize()`
is applied afterwards by the generation script, like every other class.

Ranges live in RANGES so the pilot run (Step 7) can tune them in one place.
Factor ranges skip a small band around 1.0 so no edit is a near no-op.
"""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Parameter ranges (tuned these after the pilot run)
RANGES = {
    "brightness": {"lo": 0.75, "hi": 1.30, "gap": 0.12},
    "contrast":   {"lo": 0.70, "hi": 1.40, "gap": 0.15},
    "saturation": {"lo": 0.65, "hi": 1.40, "gap": 0.15},
    "white_balance": {"min_abs": 0.06, "max_abs": 0.15},
    "crop": {"keep_lo": 0.85, "keep_hi": 0.95},
    "sharpen": {"radius": (1.0, 2.0), "percent": (50, 150), "threshold": 2},
    "jpeg": {"q_lo": 40, "q_hi": 65},
}


def _factor(rng: np.random.Generator, lo: float, hi: float, gap: float) -> float:
    """Factor below or above 1.0 (50/50), never within `gap` of 1.0."""
    if rng.random() < 0.5:
        return float(rng.uniform(lo, 1.0 - gap))
    return float(rng.uniform(1.0 + gap, hi))


# Operations
def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def sample_brightness(rng):
    return {"factor": _factor(rng, **RANGES["brightness"])}


def apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def sample_contrast(rng):
    return {"factor": _factor(rng, **RANGES["contrast"])}


def apply_saturation(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(img).enhance(factor)


def sample_saturation(rng):
    return {"factor": _factor(rng, **RANGES["saturation"])}


def apply_white_balance(img: Image.Image, shift: float) -> Image.Image:
    """Warm (shift > 0) or cool (shift < 0) the image by scaling red and blue
    in opposite directions, like a color-temperature slider."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr[..., 0] *= 1.0 + shift   # red
    arr[..., 2] *= 1.0 - shift   # blue
    return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8))


def sample_white_balance(rng):
    r = RANGES["white_balance"]
    mag = rng.uniform(r["min_abs"], r["max_abs"])
    return {"shift": float(mag if rng.random() < 0.5 else -mag)}


def apply_crop(img: Image.Image, keep: float, x_frac: float, y_frac: float) -> Image.Image:
    """Crop a window keeping `keep` of each side, positioned by x_frac/y_frac
    in [0, 1], then resize back to the original size."""
    w, h = img.size
    cw, ch = round(w * keep), round(h * keep)
    left = round((w - cw) * x_frac)
    top = round((h - ch) * y_frac)
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BICUBIC)


def sample_crop(rng):
    r = RANGES["crop"]
    return {"keep": float(rng.uniform(r["keep_lo"], r["keep_hi"])),
            "x_frac": float(rng.random()), "y_frac": float(rng.random())}


def apply_sharpen(img: Image.Image, radius: float, percent: int, threshold: int) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent,
                                              threshold=threshold))


def sample_sharpen(rng):
    r = RANGES["sharpen"]
    return {"radius": float(rng.uniform(*r["radius"])),
            "percent": int(rng.integers(r["percent"][0], r["percent"][1] + 1)),
            "threshold": int(r["threshold"])}


def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Re-save as JPEG at a lower quality (simulates an export/upload)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def sample_jpeg(rng):
    r = RANGES["jpeg"]
    return {"quality": int(rng.integers(r["q_lo"], r["q_hi"] + 1))}


# Registry
OPS = {
    "brightness":    (apply_brightness, sample_brightness),
    "contrast":      (apply_contrast, sample_contrast),
    "saturation":    (apply_saturation, sample_saturation),
    "white_balance": (apply_white_balance, sample_white_balance),
    "crop":          (apply_crop, sample_crop),
    "sharpen":       (apply_sharpen, sample_sharpen),
    "jpeg":          (apply_jpeg, sample_jpeg),
}


def apply_op(img: Image.Image, name: str, params: dict) -> Image.Image:
    return OPS[name][0](img, **params)


def sample_op(name: str, rng: np.random.Generator) -> dict:
    return OPS[name][1](rng)


def strength_of(name: str, params: dict) -> float:
    """One number per edit for the metadata `strength` column:
    how far the main parameter is from "no edit"."""
    if name in ("brightness", "contrast", "saturation"):
        return abs(params["factor"] - 1.0)
    if name == "white_balance":
        return abs(params["shift"])
    if name == "crop":
        return 1.0 - params["keep"]
    if name == "sharpen":
        return params["percent"] / 100.0
    if name == "jpeg":
        return (100 - params["quality"]) / 100.0
    raise KeyError(name)


# Recipes: one main operation + 0-2 extras, like a real touch-up

# Order the ops are applied in, like a typical editing workflow:
# framing first, then tone and color, then sharpening, then export.
APPLY_ORDER = ["crop", "brightness", "contrast", "white_balance",
               "saturation", "sharpen", "jpeg"]

N_EXTRA_PROBS = [0.4, 0.4, 0.2]


def recipe_seed(source_id: str) -> int:
    """Fixed seed per source image, so every recipe can be recreated exactly.
    The "classical:" prefix keeps it independent of the split hash."""
    return int(hashlib.sha256(f"classical:{source_id}".encode()).hexdigest()[:8], 16)


def make_recipe(source_id: str, main: str | None = None, attempt: int = 0) -> dict:
    """Pick the main op (balanced across sources unless given), 0-2 distinct
    extras, and parameters for each, all from the source's fixed seed.

    `attempt` > 0 redraws the extras and parameters (used when an edit fails
    the minimum-change check) but keeps the same main op, so the balance of
    main ops across sources is unchanged.
    """
    base_seed = recipe_seed(source_id)
    seed = base_seed if attempt == 0 else recipe_seed(f"{source_id}:retry{attempt}")
    rng = np.random.default_rng(seed)
    names = list(OPS)
    if main is None:
        main = names[base_seed % len(names)]     
    others = [n for n in names if n != main]
    n_extra = int(rng.choice(len(N_EXTRA_PROBS), p=N_EXTRA_PROBS))
    extras = [str(n) for n in rng.choice(others, size=n_extra, replace=False)]
    chosen = sorted([main] + extras, key=APPLY_ORDER.index)
    return {
        "source_id": source_id,
        "seed": seed,
        "attempt": attempt,
        "main": main,
        "ops": [{"name": n, "params": sample_op(n, rng)} for n in chosen],
    }


def apply_recipe(img: Image.Image, recipe: dict) -> Image.Image:
    for op in recipe["ops"]:
        img = apply_op(img, op["name"], op["params"])
    return img


# Ops that move pixels rather than change their values
GEOMETRIC_OPS = {"crop"}


def aligned_reference(img: Image.Image, recipe: dict) -> Image.Image:
    """The source with only the recipe's geometric ops applied.

    Pixel-aligned metrics (LPIPS, SSIM) read a crop as a huge change because
    every pixel shifts, even though the content barely changes. Comparing the
    edit against this reference measures only the tone/color/detail changes,
    the same way diffusion edits (which keep pixels in place) are measured.
    """
    for op in recipe["ops"]:
        if op["name"] in GEOMETRIC_OPS:
            img = apply_op(img, op["name"], op["params"])
    return img


def metadata_row(recipe: dict, path: str, split: str) -> dict:
    """One metadata.csv row in the contract's columns, plus `params` (the full
    recipe as JSON, so any image can be regenerated)."""
    main = recipe["main"]
    main_params = next(op["params"] for op in recipe["ops"] if op["name"] == main)
    return {
        "image_id": f"{recipe['source_id']}__classical",
        "source_id": recipe["source_id"],
        "path": path,
        "class": "classical",
        "edit_type": main,
        "strength": round(strength_of(main, main_params), 4),
        "mask_area": "",
        "editor": "pil",
        "split": split,
        "params": json.dumps(recipe["ops"]),
    }


# Quick visual checks (run from the repo root):
#   one image, every op:   python -m src.data.classical <image>
#   recipes on N sources:  python -m src.data.classical --recipes 8
if __name__ == "__main__":
    import csv
    import sys
    from pathlib import Path

    from src.data.preprocess import to_working_size

    out = Path("data/scratch/classical_demo")
    out.mkdir(parents=True, exist_ok=True)

    if sys.argv[1] == "--recipes":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
        with open("data/manifest_real.csv") as f:
            rows = list(csv.DictReader(f))[:n]
        for r in rows:
            base = to_working_size(Image.open(Path("data") / r["path"]))
            recipe = make_recipe(r["source_id"])
            base.save(out / f"{r['source_id']}_0_original.png")
            apply_recipe(base, recipe).save(out / f"{r['source_id']}_1_classical.png")
            ops = " + ".join(op["name"] for op in recipe["ops"])
            print(f"{r['source_id']}  main={recipe['main']:13s}  ops: {ops}")
    else:
        base = to_working_size(Image.open(sys.argv[1]))
        base.save(out / "00_original.png")
        rng = np.random.default_rng(0)
        for i, name in enumerate(OPS, start=1):
            params = sample_op(name, rng)
            apply_op(base, name, params).save(out / f"{i:02d}_{name}.png")
            print(f"{name:14s} strength={strength_of(name, params):.3f}  {params}")
    print(f"Saved to {out}/")