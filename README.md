# Light vs. Heavy AI Edit Detection

COMS 4776 course project. Can detectors tell **how much** an image's content was
changed, rather than **whether** AI touched it?

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Download the real images (everyone)

1. Accept the ImageNet terms on Hugging Face:
   https://huggingface.co/datasets/ILSVRC/imagenet-1k (approval is usually instant).
2. Log in with a read token from https://huggingface.co/settings/tokens:
   ```bash
   huggingface-cli login
   ```
3. Quick test, then the full download (5,000 images, about 650 MB):
   ```bash
   python scripts/download_imagenet.py --num 20
   python scripts/download_imagenet.py
   ```

The first person to run the full download commits `data/manifest_real.csv`.
After that, everyone's download is checked against it automatically. You can
also run `python scripts/download_imagenet.py --verify` at any time.

## Layout

```
configs/data.yaml            shared data settings (change only by team agreement)
scripts/download_imagenet.py real-image download + verification
src/data/contract.py         class names, metadata columns, split assignment
src/data/preprocess.py       shared resize + JPEG (apply to EVERY image, last)
src/models/                  detector registry; one file per detector
src/train/                   shared training script
src/eval/                    shared evaluation code
data/                        local only, not in git (except the manifest)
results/                     local only; log the runs you report in results.md
```

## Data rules

- **Never commit images.** Only `data/manifest_real.csv` goes in git.
- **Run `standardize()` from `preprocess.py` as the last step on every image**,
  real or edited, so the file format and size can't give away the class.
- **Splits come from `assign_split(source_id)`.** Every edit of a photo inherits
  its split.
- **Each pipeline writes a metadata CSV** with the columns in `contract.py`.

## Ownership

| Person | Detector | Pipeline piece | Analysis |
|---|---|---|---|
| A | RGB ResNet-50 (`rgb_resnet50`) | Classical edits, preprocessing | Grad-CAM |
| B | DCT ResNet-50 (`dct_resnet50`) | Diffusion edits, VAE control, held-out editor | Frequency spectra |
| C | CLIP probe (`clip_probe`), off-the-shelf baselines | Evaluation code | Feature embeddings |
