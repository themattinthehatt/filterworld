"""Orchestrates the full processing pipeline."""

import logging
from pathlib import Path

from filterworld.canvas.canvas import Canvas
from filterworld.config import load_config
from filterworld.filters.base import Filter
from filterworld.filters.dinov1_filter import DINOv1Filter
from filterworld.filters.dinov2_filter import DINOv2Filter
from filterworld.filters.file_filter import FileFilter
from filterworld.filters.identity_filter import IdentityFilter
from filterworld.media.video import VideoReader
from filterworld.writers.video_writer import VideoWriter

logger = logging.getLogger(__name__)

# file extensions that indicate pre-computed filter output
_FILE_FILTER_EXTENSIONS = {'.json', '.jsonl', '.pkl', '.pickle', '.npz', '.pt', '.csv'}

# standardized model name -> (huggingface identifier, filter class)
_MODEL_REGISTRY: dict[str, tuple[str, type[Filter]]] = {
    'dinov1-small': ('facebook/dino-vits16', DINOv1Filter),
    'dinov1-base': ('facebook/dino-vitb16', DINOv1Filter),
    'dinov2-small': ('facebook/dinov2-small', DINOv2Filter),
    'dinov2-base': ('facebook/dinov2-base', DINOv2Filter),
}


def _is_file_filter_path(model_path: str) -> bool:
    """Determine whether model_path points to a pre-computed output file.

    Args:
        model_path: model path or identifier provided by the user

    Returns:
        True if the path looks like a pre-computed output file
    """
    p = Path(model_path)
    return p.suffix in _FILE_FILTER_EXTENSIONS and p.exists()


def build_filter(model_path: str, resolution: int | None = None) -> Filter:
    """Dispatch to the appropriate Filter based on model_path.

    Accepts standardized model names (e.g. 'dinov1-small', 'dinov2-base')
    as well as 'identity' and pre-computed file paths.

    Args:
        model_path: model name, identifier, or pre-computed output file
        resolution: optional input resolution override passed to the filter

    Returns:
        an initialized Filter instance

    Raises:
        ValueError: if model_path is not recognized
    """
    if model_path == 'identity':
        logger.info('using identity filter (passthrough)')
        return IdentityFilter()
    if _is_file_filter_path(model_path):
        logger.info('using pre-computed filter output from %s', model_path)
        return FileFilter(model_path)
    if model_path in _MODEL_REGISTRY:
        hf_name, filter_cls = _MODEL_REGISTRY[model_path]
        logger.info('using model %s (%s)', model_path, hf_name)
        return filter_cls(hf_name, resolution=resolution)
    model_names = ', '.join(sorted(_MODEL_REGISTRY.keys()))
    raise ValueError(
        f'unsupported model: {model_path!r}. '
        f'available models: {model_names}'
    )


class Pipeline:
    """Wires together VideoReader, Filter, Canvas, and Writer.

    Args:
        video_path: path to the input video file
        model_path: model path/identifier or pre-computed output file
        config_path: optional path to a YAML config file
        output_path: path for the output video
    """

    def __init__(
        self,
        video_path: str,
        model_path: str,
        config_path: str | None,
        output_path: str,
        resolution: int | None = None,
    ) -> None:
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path
        self.resolution = resolution
        self.config = load_config(config_path)

    def run(self) -> None:
        """Execute the pipeline: read frames, filter, render, write."""
        logger.info('starting pipeline: %s -> %s', self.video_path, self.output_path)

        reader = VideoReader(self.video_path)
        vid_filter = build_filter(self.model_path, resolution=self.resolution)
        canvas = Canvas(self.config)

        output_cfg = self.config.output
        fps = output_cfg.fps or reader.fps
        writer = VideoWriter(self.output_path, fps=fps, fourcc=output_cfg.codec)

        try:
            for frame in reader:
                filter_output = vid_filter.process_frame(frame)
                rendered = canvas.render(frame, filter_output)
                writer.write_frame(rendered)
        finally:
            writer.close()

        logger.info('pipeline complete: %s', self.output_path)
