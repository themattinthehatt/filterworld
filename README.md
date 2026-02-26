# filterworld

Visualize computer vision model outputs on video.

## Installation

Create and activate a conda environment:

```bash
conda create -n filter python=3.12
conda activate filter
```

Install dependencies:

```bash
pip install numpy opencv-python transformers torch scikit-learn
```

Install filterworld in development mode:

```bash
pip install -e .
```

## Quick start

Run the identity filter (passthrough) on a video to verify the pipeline works:

```bash
filterworld run input.mp4 identity -o output.mp4
```

This reads `input.mp4`, passes each frame through unchanged, and writes the result to `output.mp4`. If `-o` is omitted the output defaults to `input_filtered.mp4`.

### Using a config file

Use `--config` to control the output format. For example, to downscale the output:

```yaml
# small.yaml
layout:
  type: grid
  rows: 1
  cols: 1

panes: []

output:
  fps: null
  codec: mp4v
  width: 320
  height: 240
```

```bash
filterworld run input.mp4 identity --config small.yaml -o small_output.mp4
```

See `configs/default.yaml` for the full default configuration.

### Feature extraction

Feature visualization is a two-step workflow: first precompute PCA weights, then render.

Available models:
- `dinov1-small` — DINO ViT-S/16
- `dinov1-base` — DINO ViT-B/16
- `dinov2-small` — DINOv2 ViT-S/14
- `dinov2-base` — DINOv2 ViT-B/14
- `dinov3-small` — DINOv3 ViT-S/16 (gated, requires HuggingFace auth)
- `dinov3-base` — DINOv3 ViT-B/16 (gated, requires HuggingFace auth)
- `vitmae-base` — ViT-MAE ViT-B/16

DINOv3 models are gated on HuggingFace. Run `hf auth login` and request access to the model page before use. If authentication is missing, filterworld will print detailed setup instructions.

All models support `--resolution N` to override the default input size (e.g. `--resolution 448` for higher-resolution feature maps).

**Step 1: Precompute PCA weights**

```bash
filterworld precompute input.mp4 dinov2-small -o dino_pca.npz
```

This runs the model on a subset of frames (default 200), fits a 3-component PCA across all patch embeddings, and saves the projection weights to `dino_pca.npz`.

Options:
- `--max-frames N` — maximum number of frames to sample for PCA fitting (default: 200)

**Step 2: Render with PCA features**

```bash
filterworld run input.mp4 dinov2-small --config configs/dino.yaml -o dino_output.mp4
```

The included `configs/dino.yaml` sets up a two-column grid with the original frame on the left and PCA-colored DINO features on the right. The `pca_path` in the config should point to the `.npz` file generated in step 1.
