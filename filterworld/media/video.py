"""Video reading and frame extraction."""

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Reads frames from a video file.

    Wraps OpenCV's VideoCapture to provide an iterable interface over
    video frames as numpy arrays (H, W, C) in RGB order.

    Args:
        video_path: path to the input video file

    Raises:
        FileNotFoundError: if video_path does not exist
        RuntimeError: if the video file cannot be opened
    """

    def __init__(self, video_path: str) -> None:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f'video not found: {video_path}')

        self._path = path
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f'failed to open video: {video_path}')

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            'opened video %s: %dx%d, %.2f fps, %d frames',
            path.name, self._width, self._height, self._fps, self._frame_count,
        )

    @property
    def fps(self) -> float:
        """Frames per second of the video."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total number of frames in the video."""
        return self._frame_count

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return self._height

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over video frames.

        Yields:
            numpy array of shape (H, W, 3) in RGB order, dtype uint8
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            # opencv reads BGR; convert to RGB
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        """Return the total number of frames."""
        return self._frame_count

    def __del__(self) -> None:
        """Release the video capture resource."""
        if hasattr(self, '_cap') and self._cap is not None:
            self._cap.release()
