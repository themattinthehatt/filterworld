"""Layer base class."""

from abc import ABC, abstractmethod

import numpy as np

from filterworld.filters.base import FilterOutput


class Layer(ABC):
    """Base class for visual layers rendered onto a pane.

    A Layer takes a frame and filter output and draws one visual element
    (e.g. the raw image, bounding boxes, segmentation masks) onto a
    target image.

    Args:
        opacity: layer opacity in [0.0, 1.0], default 1.0
    """

    def __init__(self, opacity: float = 1.0) -> None:
        self.opacity = np.clip(opacity, 0.0, 1.0)

    @abstractmethod
    def render(
        self,
        target: np.ndarray,
        frame: np.ndarray,
        filter_output: FilterOutput,
    ) -> np.ndarray:
        """Render this layer onto the target image.

        Args:
            target: current composited image of shape (H, W, 3), dtype uint8
            frame: original video frame of shape (H, W, 3), dtype uint8
            filter_output: output from the filter for this frame

        Returns:
            updated target image of shape (H, W, 3), dtype uint8
        """
        ...
