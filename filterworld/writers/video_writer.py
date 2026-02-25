"""Writes rendered frames to mp4."""

import logging
from pathlib import Path

import cv2
import numpy as np

from filterworld.writers.base import Writer

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 30.0


class VideoWriter(Writer):
    """Writes frames to an mp4 file using OpenCV.

    The underlying cv2.VideoWriter is initialized lazily on the first
    call to `write_frame`, so frame dimensions do not need to be known
    at construction time.

    Args:
        output_path: path for the output video file
        fps: frames per second for the output video
        fourcc: four-character codec code (default 'mp4v')
    """

    def __init__(
        self,
        output_path: str,
        fps: float = _DEFAULT_FPS,
        fourcc: str = 'mp4v',
    ) -> None:
        self._output_path = Path(output_path)
        self._fps = fps
        self._fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0

    def _ensure_writer(self, height: int, width: int) -> None:
        """Lazily create the cv2.VideoWriter on first frame.

        Args:
            height: frame height in pixels
            width: frame width in pixels
        """
        if self._writer is not None:
            return

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self._fourcc)
        self._writer = cv2.VideoWriter(
            str(self._output_path), fourcc, self._fps, (width, height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f'failed to open video writer: {self._output_path}')

        logger.info(
            'writing video to %s (%dx%d @ %.2f fps)',
            self._output_path, width, height, self._fps,
        )

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single rendered frame.

        Args:
            frame: image array of shape (H, W, 3), dtype uint8, RGB order
        """
        h, w = frame.shape[:2]
        self._ensure_writer(h, w)
        # convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)
        self._frame_count += 1

    def close(self) -> None:
        """Release the video writer."""
        if self._writer is not None:
            self._writer.release()
            logger.info(
                'wrote %d frames to %s', self._frame_count, self._output_path,
            )
            self._writer = None

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count
