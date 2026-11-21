"""Shared preprocessing. Every image of every class goes through `standardize`.

Without this, detectors learn shortcuts such as "PNG = fake", or "512px = fake"
instead of anything about the edit.
"""

import io

from PIL import Image


def resize_center_crop(img: Image.Image, size: int = 256) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))),
                     Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def jpeg_roundtrip(img: Image.Image, quality: int = 90) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def standardize(img: Image.Image, size: int = 256, jpeg_quality: int = 90) -> Image.Image:
    """Resize + crop to a square, then JPEG re-encode. Apply as the LAST step."""
    return jpeg_roundtrip(resize_center_crop(img, size), jpeg_quality)
