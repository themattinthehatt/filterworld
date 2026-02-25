# filterworld — Design Document

## Overview

`filterworld` is a Python package for visualizing the outputs of computer vision models on video. Given an input video and a model (or pre-computed model output file), it produces a rendered output video showing the model's outputs — feature maps, bounding boxes, segmentation masks, depth maps, keypoints, and more.

The initial interface is a command-line tool. A future goal is an interactive viewer (scrollable, frame-seekable), and the architecture is designed from the start to accommodate this without a rewrite.

---

## Goals

- Visualize diverse model outputs on short mp4 videos
- Support models from Hugging Face `transformers` as a first-class integration, with room for others (SAM, Lightning Pose, Video Depth Anything, etc.)
- Allow flexible visualization layouts: overlays, side-by-side panels, multi-pane arrangements
- Accept either a live model or a pre-computed output file as input
- Be easy to extend: new model types, new layer types, new output targets should each require minimal changes elsewhere
- Lay groundwork for an interactive viewer without prematurely building it

---

## Repository Structure

```
filterworld/                      # top-level repo directory
├── filterworld/                  # main package
│   ├── __init__.py
│   ├── cli/                      # command-line interface
│   │   ├── __init__.py
│   │   └── main.py
│   ├── filters/                  # model wrappers and output loaders
│   │   ├── __init__.py
│   │   ├── base.py               # Filter base class and FilterOutput dataclasses
│   │   ├── file_filter.py        # loads pre-computed outputs from disk
│   │   └── hf_filter.py          # wraps Hugging Face transformers models
│   ├── media/                    # media ingestion
│   │   ├── __init__.py
│   │   └── video.py              # VideoReader: reads frames from mp4
│   ├── layers/                   # rendering primitives
│   │   ├── __init__.py
│   │   ├── base.py               # Layer base class
│   │   ├── image_layer.py        # renders a raw image/frame
│   │   ├── feature_layer.py      # renders CNN/ViT feature maps
│   │   ├── bbox_layer.py         # renders bounding boxes
│   │   ├── segmentation_layer.py # renders segmentation masks
│   │   ├── depth_layer.py        # renders depth maps
│   │   └── keypoint_layer.py     # renders keypoints and skeleton
│   ├── canvas/                   # assembles layers into Panes and frames
│   │   ├── __init__.py
│   │   ├── pane.py               # Pane: one panel in the final frame
│   │   └── canvas.py             # Canvas: assembles Panes into a full frame
│   └── writers/                  # output targets
│   │   ├── __init__.py
│   │   ├── base.py               # Writer base class
│   │   └── video_writer.py       # writes rendered frames to mp4
│   └── pipeline.py               # orchestrates the whole process, CLI utilizes this object
├── tests/
│   ├── conftest.py
│   ├── test_filters/
│   ├── test_layers/
│   ├── test_canvas/
│   └── test_writers/
├── pyproject.toml
├── README.md
└── DESIGN.md                     # this file
```

---

## Subpackage Responsibilities

### `cli`

Entry point for the command-line interface. Parses arguments, constructs the pipeline (media → filter → layers → canvas → writer), and runs it.

**Initial interface:**
```
filterworld filter <video> <model_path_or_output_file> [--config config.yaml]
```

The `--config` flag points to an optional YAML/TOML file that specifies visualization parameters (colormap, which feature layer indices to show, keypoint skeleton connectivity, layout, etc.). This makes runs reproducible and avoids an explosion of CLI flags.

The CLI should be a thin orchestration layer. It should not contain business logic — it calls into `filters`, `canvas`, and `writers`.

---

### `media`

Responsible for reading input media and producing a stream of frames.

**`VideoReader`** (`media/video.py`):
- Wraps an mp4 file and yields frames (as numpy arrays or tensors) on demand
- Exposes metadata: fps, frame count, width, height
- Supports random frame access (needed for the future interactive case)
- Does not perform any processing — pure ingestion

Future modules in this subpackage: `StreamReader` (live camera), `SimulatedReader` (synthetic data).

---

### `filters`

Responsible for producing structured model outputs from either a live model or a pre-computed file. Downstream code (layers, canvas) should be entirely agnostic to which source was used.

**`FilterOutput`** (`filters/base.py`):
A family of dataclasses (or a single dataclass with optional fields) that standardizes model output. All filter types produce one of these per frame. Examples:

```python
@dataclass
class BBoxOutput:
    boxes: np.ndarray        # (N, 4) in xyxy or xywh
    scores: np.ndarray       # (N,)
    labels: list[str]        # (N,)

@dataclass
class SegmentationOutput:
    masks: np.ndarray        # (N, H, W) boolean or float
    labels: list[str]

@dataclass
class FeatureOutput:
    feature_maps: list[np.ndarray]   # one per requested layer
    layer_names: list[str]

@dataclass
class KeypointOutput:
    keypoints: np.ndarray    # (N_instances, K, 2 or 3)
    scores: np.ndarray
    skeleton: list[tuple]    # connectivity for drawing edges

@dataclass
class DepthOutput:
    depth_map: np.ndarray    # (H, W) float
```

**`Filter`** (abstract base class, `filters/base.py`):
```python
class Filter(ABC):
    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> FilterOutput:
        ...

    def process_video(self, reader: VideoReader) -> Iterator[FilterOutput]:
        # default: calls process_frame for each frame
        ...
```

**`HuggingFaceFilter`** (`filters/hf_filter.py`):
- Accepts a model path or identifier for a `transformers` model
- Handles model loading, preprocessing, inference, and postprocessing into a `FilterOutput`
- Subclasses per task type: `HFDetectionFilter`, `HFSegmentationFilter`, `HFDepthFilter`, `HFFeatureFilter`

**`FileFilter`** (`filters/file_filter.py`):
- Loads pre-computed outputs from disk (e.g., HDF5, JSON, npz)
- Produces the same `FilterOutput` types as live model filters
- Enables testing and visualization without re-running inference

---

### `layers`

Rendering primitives. Each `Layer` takes a frame and/or a `FilterOutput` and returns a rendered image (numpy array, same shape as the output frame).

**`Layer`** (abstract base class, `layers/base.py`):
```python
class Layer(ABC):
    @abstractmethod
    def render(self, frame: np.ndarray | None, output: FilterOutput | None) -> np.ndarray:
        ...
```

Layers are stateless with respect to the pipeline — they do not hold references to video readers or filters. Each call to `render` is self-contained.

**Planned layer types:**
- `ImageLayer` — renders a raw input frame, optionally with transforms (resize, normalize for display)
- `FeatureLayer` — renders a single feature map from a `FeatureOutput`; accepts colormap, channel aggregation strategy (mean, PCA, specific channel index)
- `BBoxLayer` — draws bounding boxes and labels on an image
- `SegmentationLayer` — overlays segmentation masks on an image with configurable opacity and colormap
- `DepthLayer` — renders a depth map with configurable colormap
- `KeypointLayer` — draws keypoints and skeleton edges on an image
- `TextLayer` — overlays text (frame number, timestamps, metadata) on an image

Layers that produce overlays (BBox, Segmentation, Depth, Keypoint, Text) should accept a `background` image to composite onto. Layers that produce standalone images (Image, Feature) return a full image.

---

### `canvas`

Assembles layers into `Pane` objects, and assembles `Pane` objects into a full output frame. This is where layout logic lives.

**`Pane`** (`canvas/pane.py`):
- Holds an ordered list of `Layer` objects
- Renders them in order, compositing the results (later layers draw on top of earlier ones)
- Has a fixed output size (width, height)
- Is agnostic to where it appears in the final frame

```python
class Pane:
    layers: list[Layer]
    width: int
    height: int

    def render(self, frame: np.ndarray, output: FilterOutput) -> np.ndarray:
        # render layers in sequence, compositing onto a blank canvas
        ...
```

**`Canvas`** (`canvas/canvas.py`):
- Holds an ordered list of `Pane` objects and a layout specification (e.g., 1 row × 2 columns, grid, etc.)
- Assembles panes into a final frame image
- Returns a single numpy array representing the full output frame

```python
class Canvas:
    panes: list[Pane]
    layout: Layout          # e.g., GridLayout(rows=1, cols=2)
    output_size: tuple      # (width, height)

    def render(self, frame: np.ndarray, output: FilterOutput) -> np.ndarray:
        ...
```

Layout objects (a `Layout` base class with subclasses like `GridLayout`, `OverlayLayout`) specify how panes are sized and positioned within the output frame. This allows side-by-side panels, picture-in-picture, single-pane, etc.

---

### `writers`

Responsible for consuming rendered frames and emitting them to an output target.

**`Writer`** (abstract base class, `writers/base.py`):
```python
class Writer(ABC):
    @abstractmethod
    def write_frame(self, frame: np.ndarray) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self): ...
    def __exit__(self, *args): ...
```

**`VideoWriter`** (`writers/video_writer.py`):
- Wraps `cv2.VideoWriter` or `imageio`
- Accepts output path, fps, codec
- Writes rendered frames sequentially to an mp4

**Future writers:** `StreamWriter` for pushing frames to an interactive viewer, `DisplayWriter` for showing frames in a window during development.

---

## Data Flow

```
VideoReader
    │ frame (np.ndarray)
    ▼
Filter.process_frame()
    │ FilterOutput
    ▼
Canvas.render(frame, output)
    │  calls each Pane.render()
    │    each Pane calls its Layer.render() in sequence
    │
    │ rendered frame (np.ndarray)
    ▼
Writer.write_frame()
    │
    ▼
output.mp4
```

The pipeline loop (in `cli` or a `Pipeline` helper class) iterates over frames from the `VideoReader`, passing each frame through the filter and then through the canvas, and writing the result.

---

## Configuration

Visualization parameters are specified via an optional YAML config file (`--config`). This covers:

- Which feature layer indices to visualize
- Colormap selection per layer
- Overlay opacity
- Keypoint skeleton connectivity
- Canvas layout (number of panes, arrangement)
- Output video codec and fps

Example:
```yaml
layout:
  type: grid
  rows: 1
  cols: 2

panes:
  - layers:
      - type: image
  - layers:
      - type: feature
        layer_index: 11
        colormap: viridis
        aggregation: pca

output:
  fps: 30
  codec: mp4v
```

---

## Extension Points

| To add...                  | Where to extend                          |
|----------------------------|------------------------------------------|
| A new model type           | New `Filter` subclass in `filters/`      |
| A new visualization style  | New `Layer` subclass in `layers/`        |
| A new layout               | New `Layout` subclass in `canvas/`       |
| A new output target        | New `Writer` subclass in `writers/`      |
| A new media source         | New reader class in `media/`             |

---

## Future: Interactive Viewer

The architecture is designed to support an interactive viewer (frame-seekable, real-time scrubbing) without restructuring. The key properties that enable this:

- `VideoReader` supports random frame access by index
- `Filter` has a `process_frame(frame, frame_idx)` interface (not just a sequential iterator)
- `Canvas` is a pure function of `(frame, output) → rendered_frame` with no internal state
- `Writer` is behind an abstract interface — a `StreamWriter` would push frames to a viewer instead of a file

When implementing the interactive viewer, the main addition will be a new `Writer` subclass (or a separate viewer loop) that requests specific frames by index rather than iterating sequentially. The `filters`, `layers`, and `canvas` subpackages require no changes.

---

## Development Priorities

1. **Phase 1 — Core pipeline:** `VideoReader`, `Filter` base + `FileFilter`, `ImageLayer`, `FeatureLayer`, `Canvas` with `GridLayout`, `VideoWriter`, CLI skeleton
2. **Phase 2 — Model integration:** `HuggingFaceFilter` subclasses for detection, segmentation, depth, features
3. **Phase 3 — Full layer set:** `BBoxLayer`, `SegmentationLayer`, `DepthLayer`, `KeypointLayer`, `TextLayer`
4. **Phase 4 — Config system:** YAML config parsing, layout customization
5. **Phase 5 — Interactive viewer:** `StreamWriter`, frame-seeking UI

---

## Dependencies (anticipated)

- `opencv-python` — video I/O, drawing primitives
- `numpy` — array manipulation throughout
- `torch` / `transformers` — model inference
- `typer` or `click` — CLI
- `pyyaml` — config parsing
- `imageio` or `ffmpeg-python` — optional alternative video I/O
- `matplotlib` — colormaps
