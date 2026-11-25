"""Generate the diffusion edits and the VAE-only control (Person B).

For each of the first `num_sources` usable real images in data/manifest_real.csv
(short side >= min_short_side, same filter as the classical pipeline), makes one
version per entry in the config's `edits`: VAE round-trip, light/heavy img2img,
InstructPix2Pix lighting edit, large-mask inpainting.

Same pipeline as every other class:
    to_working_size() -> edit -> save_standardized() (one JPEG encode)

Outputs (under data_root):
  <out_dir>/<image_id>.jpg         standardized image
  <out_dir>/masks/<image_id>.png   inpainting mask at the standardized size
  <out_dir>/config.yaml            the config used
  <metadata>                       one row per image, contract columns

Matched set (config `matched`; evaluation only, not listed in train.yaml):
every source also gets the IP2P instructions it did not draw above, so each
source has all of them. scripts/make_matched_classical.py makes the same
lighting edits with classical tools; pairs join on (source_id, edit_type,
direction).
  <matched.out_dir>/<source_id>__ip2p_<edit_type>_<direction>.jpg
  <matched.metadata>               contract columns + direction, one row per
                                   (source, instruction). The row for the
                                   instruction drawn above points at that
                                   existing image instead of a new one.

IP2P check (config edits.ip2p: max_lpips, require_direction, max_attempts):
InstructPix2Pix sometimes overshoots (strong color casts, changed content) or
moves the lighting the wrong way. Like the classical pipeline's LPIPS filter,
each IP2P result is scored against its source on the standardized images and
redrawn with a new seed if it fails; if no attempt passes, the best one is kept
and marked passed=false in `params`. The outcome (seed, attempt, lpips, shift,
passed) is also appended to <out_dir>/ip2p_checks.jsonl so a restarted run
writes the same metadata.

Existing images are skipped, so an interrupted run can be restarted; the
metadata is rewritten in full each run (parameters are re-drawn from the seed).

Usage (GPU node):
  python scripts/generate_diffusion.py
  python scripts/generate_diffusion.py --num-sources 3   # smoke test
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.contract import METADATA_COLUMNS, assign_split  # noqa: E402
from src.data.preprocess import save_standardized, standardize, to_working_size  # noqa: E402


def image_seed(seed: int, image_id: str) -> int:
    """Per-image seed, independent of processing order."""
    return int(hashlib.sha256(f"diffusion:{seed}:{image_id}".encode()).hexdigest()[:8], 16)


def imagenet_names() -> list[str]:
    from torchvision.models import ResNet50_Weights
    return ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def load_pipelines(models: dict, device: str) -> dict:
    from diffusers import (StableDiffusionImg2ImgPipeline,
                           StableDiffusionInpaintPipeline,
                           StableDiffusionInstructPix2PixPipeline)

    def load(cls, repo):
        # The safety checker blacks out false positives, which would add a
        # spurious all-black mode to the AI classes.
        kw = dict(torch_dtype=torch.float16, safety_checker=None,
                  requires_safety_checker=False)
        try:
            pipe = cls.from_pretrained(repo, variant="fp16", **kw)
        except (OSError, ValueError):  # repo has no fp16 variant files
            pipe = cls.from_pretrained(repo, **kw)
        pipe.set_progress_bar_config(disable=True)
        return pipe.to(device)

    return {
        "img2img": load(StableDiffusionImg2ImgPipeline, models["sd15"]),
        "inpaint": load(StableDiffusionInpaintPipeline, models["sd15_inpaint"]),
        "ip2p": load(StableDiffusionInstructPix2PixPipeline, models["ip2p"]),
    }


def rect_mask(rng: np.random.Generator, size: int, area_range, aspect_range):
    """Axis-aligned rectangle with a random target area, aspect ratio and position.
    Returns (mask image, bbox as fractions [x0, y0, x1, y1])."""
    area = rng.uniform(*area_range)
    aspect = rng.uniform(*aspect_range)
    w = min(1.0, np.sqrt(area * aspect))
    h = min(1.0, np.sqrt(area / aspect))
    x0, y0 = rng.uniform(0, 1 - w), rng.uniform(0, 1 - h)
    m = np.zeros((size, size), dtype=np.uint8)
    m[round(y0 * size):round((y0 + h) * size), round(x0 * size):round((x0 + w) * size)] = 255
    bbox = [round(float(v), 4) for v in (x0, y0, x0 + w, y0 + h)]
    return Image.fromarray(m), bbox


def ip2p_edit(ins: dict, spec: dict, models: dict):
    """(row fields, params) for one InstructPix2Pix lighting instruction."""
    return (dict(edit_type=ins["edit_type"], editor="ip2p"),
            dict(model=models["ip2p"], direction=ins["direction"], instruction=ins["text"],
                 num_inference_steps=spec["num_inference_steps"],
                 guidance_scale=spec["guidance_scale"],
                 image_guidance_scale=spec["image_guidance_scale"]))


def draw_params(name: str, spec: dict, cfg: dict, rng: np.random.Generator, size: int,
                prompt: str):
    """Everything random about one edit, drawn before (and independently of) running it.

    Returns (row fields, params for the `params` column, mask or None). Drawn
    even when the image already exists, so a resumed run writes the same
    metadata as a fresh one.
    """
    steps, gs = cfg["num_inference_steps"], cfg["guidance_scale"]
    models = cfg["models"]
    if name == "vae":
        return (dict(edit_type="vae", editor="sd15"),
                dict(model=models["sd15"], latent="mode"), None)
    if name.startswith("img2img"):
        s = round(float(rng.uniform(*spec["strength"])), 4)
        return (dict(edit_type="img2img", editor="sd15", strength=s),
                dict(model=models["sd15"], prompt=prompt, strength=s,
                     num_inference_steps=steps, guidance_scale=gs), None)
    if name == "ip2p":
        ins = spec["instructions"][int(rng.integers(len(spec["instructions"])))]
        return (*ip2p_edit(ins, spec, models), None)
    if name == "inpaint":
        mask, bbox = rect_mask(rng, size, spec["mask_area"], spec["mask_aspect"])
        area = round(float((np.asarray(mask) > 127).mean()), 4)
        return (dict(edit_type="inpaint", editor="sd15", mask_area=area),
                dict(model=models["sd15_inpaint"], prompt=prompt, mask_bbox=bbox,
                     num_inference_steps=steps, guidance_scale=gs), mask)
    raise KeyError(f"No generator for edit {name!r}")


@torch.no_grad()
def vae_roundtrip(vae, img: Image.Image, device: str) -> Image.Image:
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)[None].to(device, vae.dtype)
    x = x / 127.5 - 1
    z = vae.encode(x).latent_dist.mode()
    y = vae.decode(z).sample.clamp(-1, 1)
    y = ((y[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    return Image.fromarray(y)


def run_edit(name: str, p: dict, mask, pipes: dict, src: Image.Image,
             gen: torch.Generator, device: str) -> Image.Image:
    if name == "vae":
        return vae_roundtrip(pipes["img2img"].vae, src, device)
    if name.startswith("img2img"):
        return pipes["img2img"](p["prompt"], image=src, strength=p["strength"],
                                num_inference_steps=p["num_inference_steps"],
                                guidance_scale=p["guidance_scale"], generator=gen).images[0]
    if name == "ip2p":
        return pipes["ip2p"](p["instruction"], image=src,
                             num_inference_steps=p["num_inference_steps"],
                             guidance_scale=p["guidance_scale"],
                             image_guidance_scale=p["image_guidance_scale"],
                             generator=gen).images[0]
    if name == "inpaint":
        return pipes["inpaint"](p["prompt"], image=src, mask_image=mask, height=src.height,
                                width=src.width, num_inference_steps=p["num_inference_steps"],
                                guidance_scale=p["guidance_scale"], generator=gen).images[0]
    raise KeyError(f"No generator for edit {name!r}")


def lighting_shift(src: np.ndarray, out: np.ndarray, edit_type: str, direction: str) -> float:
    """How far the image moved in the instructed direction, in 0-255 pixel units
    (positive = the right way). brightness: mean of all channels; white_balance:
    mean of red minus blue, the axis the classical white-balance edit moves."""
    def level(a):
        a = a.astype(np.float64)
        return a.mean() if edit_type == "brightness" else (a[..., 0] - a[..., 2]).mean()
    return float((level(out) - level(src)) * (1 if direction == "up" else -1))


def ip2p_checked(p: dict, spec: dict, seeds: list[int], pipes: dict, src: Image.Image,
                 lpips_model, pre: dict, device: str):
    """Runs one IP2P instruction, redrawing with the next seed while the result
    fails the check. Returns (image, record); the record goes into `params`."""
    max_lpips = spec.get("max_lpips", float("inf"))
    need_direction = spec.get("require_direction", False)

    def to_t(im):
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0)
        return x.permute(2, 0, 1)[None].to(device)

    src_std = standardize(src, pre["size"], pre["jpeg_quality"])
    best = None
    for attempt, seed in enumerate(seeds):
        img = run_edit("ip2p", p, None, pipes, src, torch.Generator(device).manual_seed(seed),
                       device)
        out_std = standardize(img, pre["size"], pre["jpeg_quality"])
        with torch.no_grad():
            lp = float(lpips_model(to_t(src_std), to_t(out_std)).item())
        shift = lighting_shift(np.asarray(src_std), np.asarray(out_std),
                               spec_edit_type(p, spec), p["direction"])
        ok = lp <= max_lpips and (shift > 0 or not need_direction)
        rec = dict(seed=seed, attempt=attempt, lpips=round(lp, 4), shift=round(shift, 2),
                   passed=ok)
        rank = (need_direction and shift <= 0, lp)   # right direction first, then closest
        if best is None or rank < best[0]:
            best = (rank, img, rec)
        if ok:
            break
    return best[1], best[2]


def spec_edit_type(p: dict, spec: dict) -> str:
    return next(i["edit_type"] for i in spec["instructions"] if i["text"] == p["instruction"])


def write_metadata(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "diffusion.yaml"))
    ap.add_argument("--num-sources", type=int, help="override num_sources (smoke tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_cfg = yaml.safe_load(open(ROOT / cfg["data_config"]))
    if args.num_sources is not None:
        cfg["num_sources"] = args.num_sources
    data_root = ROOT / data_cfg["data_root"]
    pre, splits = data_cfg["preprocess"], data_cfg["splits"]
    working = pre.get("working_size", 512)
    min_side = pre.get("min_short_side", 256)
    out_dir = data_root / cfg["out_dir"]
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"diffusion": cfg, "data": data_cfg}, open(out_dir / "config.yaml", "w"),
                   sort_keys=False)

    with open(data_root / "manifest_real.csv") as f:
        sources = [r for r in csv.DictReader(f)
                   if min(int(r["width"]), int(r["height"])) >= min_side][:cfg["num_sources"]]
    names = imagenet_names()
    device = "cuda"
    pipes = load_pipelines(cfg["models"], device)

    matched = cfg.get("matched")
    if matched:
        matched_dir = data_root / matched["out_dir"]
        matched_dir.mkdir(parents=True, exist_ok=True)

    def make_row(image_id, s, path, cls, fields, params):
        return {
            "image_id": image_id,
            "source_id": s["source_id"],
            "path": str(path.relative_to(data_root)),
            "class": cls,
            "strength": "",
            "mask_area": "",
            "split": assign_split(s["source_id"], splits["train"], splits["val"]),
            "params": json.dumps(params),
            **fields,
        }

    ip2p_spec = cfg["edits"].get("ip2p", {})
    checks_path = out_dir / "ip2p_checks.jsonl"
    checks = {}
    if checks_path.exists():
        checks = {r["image_id"]: r for r in map(json.loads, open(checks_path))}
    checks_log = open(checks_path, "a")
    lpips_model = None

    def generate(name, p, mask, src, seed, path):
        """Makes the image unless it exists. Returns what goes in `params`."""
        nonlocal lpips_model
        if name != "ip2p":
            if not path.exists():
                gen = torch.Generator(device).manual_seed(seed)
                img = run_edit(name, p, mask, pipes, src, gen, device)
                save_standardized(img, path, pre["size"], pre["jpeg_quality"])
            return {"seed": seed, **p}
        image_id = path.stem
        if path.exists():
            # No log entry = made before the check existed: first seed, unchecked.
            rec = checks.get(image_id, {"seed": seed, "attempt": 0})
        else:
            if lpips_model is None:
                import lpips
                lpips_model = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
            seeds = [seed] + [image_seed(cfg["seed"], f"{image_id}:retry{k}")
                              for k in range(1, ip2p_spec.get("max_attempts", 1))]
            img, rec = ip2p_checked(p, ip2p_spec, seeds, pipes, src, lpips_model, pre, device)
            save_standardized(img, path, pre["size"], pre["jpeg_quality"])
            checks_log.write(json.dumps({"image_id": image_id, **rec}) + "\n")
            checks_log.flush()
        rec = {k: v for k, v in rec.items() if k != "image_id"}
        return {"seed": rec.pop("seed"), **p, **rec}

    rows, matched_rows = [], []
    for s in tqdm(sources, desc="sources"):
        src = to_working_size(Image.open(data_root / s["path"]), working)
        prompt = cfg["prompt"].format(label=names[int(s["imagenet_label"])])
        drawn = {}                       # edit name -> (row, params) for this source
        for name, spec in cfg["edits"].items():
            image_id = f"{s['source_id']}__{name}"
            path = out_dir / f"{image_id}.jpg"
            seed = image_seed(cfg["seed"], image_id)
            fields, p, mask = draw_params(name, spec, cfg, np.random.default_rng(seed),
                                          working, prompt)
            params = generate(name, p, mask, src, seed, path)
            if mask is not None:
                mask_path = out_dir / "masks" / f"{image_id}.png"
                if not mask_path.exists():
                    mask.resize((pre["size"], pre["size"]), Image.NEAREST).save(mask_path)
            rows.append(make_row(image_id, s, path, spec["class"], fields, params))
            drawn[name] = (rows[-1], p)

        if matched and "ip2p" in drawn:
            spec = cfg["edits"]["ip2p"]
            main_row, main_p = drawn["ip2p"]
            for ins in spec["instructions"]:
                key = (ins["edit_type"], ins["direction"])
                if key == (main_row["edit_type"], main_p["direction"]):
                    matched_rows.append({**main_row, "direction": ins["direction"]})
                    continue
                image_id = f"{s['source_id']}__ip2p_{key[0]}_{key[1]}"
                path = matched_dir / f"{image_id}.jpg"
                seed = image_seed(cfg["seed"], image_id)
                fields, p = ip2p_edit(ins, spec, cfg["models"])
                params = generate("ip2p", p, None, src, seed, path)
                matched_rows.append({**make_row(image_id, s, path, spec["class"], fields, params),
                                     "direction": ins["direction"]})

    write_metadata(data_root / cfg["metadata"], rows, METADATA_COLUMNS)
    if matched:
        write_metadata(data_root / matched["metadata"], matched_rows,
                       METADATA_COLUMNS + ["direction"])


if __name__ == "__main__":
    main()
