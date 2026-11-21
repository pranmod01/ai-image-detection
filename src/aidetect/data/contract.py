"""The data contract: class names, metadata columns and split assignment.

Every pipeline (classical edits, diffusion edits) writes rows with these
columns to its own metadata CSV, and every detector reads them. Change this file
only by team agreement.
"""

import hashlib

# Ordered by how much the content changed. The index is the ordinal label.
CLASSES = [
    "real",            # unedited ImageNet photo
    "classical",       # brightness/contrast/color/crop/sharpen/JPEG (Person A)
    "vae_only",        # SD autoencoder round-trip, no edit (Person B; control)
    "light_ai",        # low-strength img2img, InstructPix2Pix lighting (Person B)
    "heavy_ai",        # high-strength img2img, large-mask inpainting (Person B)
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

METADATA_COLUMNS = [
    "image_id",    # unique per file, e.g. "<source_id>__img2img_s0.2"
    "source_id",   # the ImageNet photo it came from (real images: image_id == source_id)
    "path",        # relative to data_root
    "class",       # one of CLASSES
    "edit_type",   # e.g. none, brightness, img2img, inpaint, ip2p, vae
    "strength",    # img2img strength / classical edit factor; empty if N/A
    "mask_area",   # fraction of pixels inpainted, 0-1; empty if N/A
    "editor",      # e.g. none, pil, sd15, sdxl, flux
    "split",       # train / val / test, from assign_split(source_id)
]


def assign_split(source_id: str, train: float = 0.70, val: float = 0.15) -> str:
    """Deterministic split from a hash of the source image id.

    All edits of one photo inherit its split, so nothing leaks across splits.
    """
    h = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < train:
        return "train"
    if h < train + val:
        return "val"
    return "test"
