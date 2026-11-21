"""Download the shared set of real ImageNet images.

Streams the first N images of the ImageNet-1k validation split from Hugging Face
and saves the original JPEG bytes (no re-encoding). Everyone who runs it gets
the same files.

One-time setup:
  1. Accept the ImageNet terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k
  2. huggingface-cli login     (paste a read token from huggingface.co/settings/tokens)

Usage:
  python scripts/download_imagenet.py               # uses configs/data.yaml
  python scripts/download_imagenet.py --num 200     # quick test
  python scripts/download_imagenet.py --verify      # check files against the manifest

The first run writes data/manifest_real.csv. Commit it. Later runs (on
teammates' machines) compare their files against it and report mismatches.
"""

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aidetect.data.contract import assign_split  # noqa: E402

MANIFEST = ROOT / "data" / "manifest_real.csv"
MANIFEST_COLS = ["image_id", "source_id", "path", "imagenet_label", "sha256",
                 "width", "height", "split"]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stream_records(hf_dataset: str, split: str):
    """Yield (image_bytes, imagenet_label) in the dataset's fixed order."""
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, split=split, streaming=True)
    ds = ds.cast_column("image", HFImage(decode=False))  # keep original bytes
    for ex in ds:
        yield ex["image"]["bytes"], ex["label"]


def save_records(records, num: int, out_dir: Path, splits: dict) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (img_bytes, label) in enumerate(tqdm(records, total=num, desc="images")):
        if i >= num:
            break
        image_id = f"in_val_{i:05d}"
        path = out_dir / f"{image_id}.JPEG"
        if not path.exists() or sha256(path.read_bytes()) != sha256(img_bytes):
            path.write_bytes(img_bytes)
        w, h = Image.open(io.BytesIO(img_bytes)).size
        rows.append({
            "image_id": image_id,
            "source_id": image_id,
            "path": str(path.relative_to(ROOT / "data")),
            "imagenet_label": label,
            "sha256": sha256(img_bytes),
            "width": w,
            "height": h,
            "split": assign_split(image_id, splits["train"], splits["val"]),
        })
    return rows


def read_manifest() -> list[dict]:
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict]) -> None:
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)


def verify(rows: list[dict] | None = None) -> bool:
    """Compare local files (or freshly downloaded rows) against the manifest."""
    expected = read_manifest()
    bad = []
    for r in expected:
        p = ROOT / "data" / r["path"]
        if not p.exists():
            bad.append((r["image_id"], "missing"))
        elif sha256(p.read_bytes()) != r["sha256"]:
            bad.append((r["image_id"], "hash mismatch"))
    if rows is not None and len(rows) != len(expected):
        print(f"WARNING: downloaded {len(rows)} images, manifest has {len(expected)}. "
              "Did someone change num_images?")
    if bad:
        print(f"FAILED: {len(bad)}/{len(expected)} images differ from the manifest:")
        for image_id, why in bad[:10]:
            print(f"  {image_id}: {why}")
        return False
    print(f"OK: all {len(expected)} images match data/manifest_real.csv")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--num", type=int, help="override num_images (for quick tests)")
    ap.add_argument("--verify", action="store_true", help="only check local files")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.verify:
        sys.exit(0 if verify() else 1)

    real = cfg["real"]
    num = args.num or real["num_images"]
    rows = save_records(stream_records(real["hf_dataset"], real["hf_split"]),
                        num, ROOT / real["out_dir"], cfg["splits"])

    if args.num is not None and args.num != real["num_images"]:
        print(f"Saved {len(rows)} images (test run; manifest not written or checked).")
        return
    if MANIFEST.exists():
        sys.exit(0 if verify(rows) else 1)
    write_manifest(rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"Saved {len(rows)} images; splits {counts}.")
    print("Wrote data/manifest_real.csv. Commit it so teammates can verify their copies.")


if __name__ == "__main__":
    main()
