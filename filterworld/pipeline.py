"""Orchestrates the full processing pipeline."""

import logging
from pathlib import Path

from filterworld.canvas.canvas import Canvas
from filterworld.config import Config, load_config
from filterworld.filters.base import Filter
from filterworld.filters.dino_filter import DINOFilter
from filterworld.filters.file_filter import FileFilter
from filterworld.filters.identity_filter import IdentityFilter
from filterworld.media.video import VideoReader
from filterworld.writers.video_writer import VideoWriter

logger = logging.getLogger(__name__)

# file extensions that indicate pre-computed filter output
_FILE_FILTER_EXTENSIONS = {'.json', '.jsonl', '.pkl', '.pickle', '.npz', '.pt', '.csv'}


def _is_file_filter_path(model_path: str) -> bool:
    """Determine whether model_path points to a pre-computed output file.

    Args:
        model_path: model path or identifier provided by the user

    Returns:
        True if the path looks like a pre-computed output file
    """
    p = Path(model_path)
    return p.suffix in _FILE_FILTER_EXTENSIONS and p.exists()


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
    ) -> None:
        self.video_path = video_path
        self.model_path = model_path
        self.output_path = output_path
        self.config = load_config(config_path)

    def _build_filter(self) -> Filter:
        """Dispatch to the appropriate Filter based on model_path.

        Returns:
            an initialized Filter instance
        """
        if self.model_path == 'identity':
            logger.info('using identity filter (passthrough)')
            return IdentityFilter()
        if _is_file_filter_path(self.model_path):
            logger.info('using pre-computed filter output from %s', self.model_path)
            return FileFilter(self.model_path)
        if 'dino' in self.model_path:
            logger.info('using DINO model %s', self.model_path)
            return DINOFilter(self.model_path)
        raise ValueError(
            f'unsupported model path: {self.model_path}. '
            f'use "identity", a file path, or a DINO model name.'
        )

    def run(self) -> None:
        """Execute the pipeline: read frames, filter, render, write."""
        logger.info('starting pipeline: %s -> %s', self.video_path, self.output_path)

        reader = VideoReader(self.video_path)
        vid_filter = self._build_filter()
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
