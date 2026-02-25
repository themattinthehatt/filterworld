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
pip install numpy opencv-python transformers
```

Install filterworld in development mode:

```bash
pip install -e .
```

## Quick start

Run the identity filter (passthrough) on a video to verify the pipeline works:

```bash
filterworld input.mp4 identity -o output.mp4
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
filterworld input.mp4 identity --config small.yaml -o small_output.mp4
```

See `filterworld/configs/default.yaml` for the full default configuration.
