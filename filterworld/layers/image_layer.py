"""Renders a raw image/frame."""

import numpy as np

from filterworld.filters.base import FilterOutput
from filterworld.layers.base import Layer


class ImageLayer(Layer):
    """Layer that renders the original video frame.

    This is typically the bottom layer in a pane's layer stack,
    providing the base image that other layers draw on top of.
    """

    def render(
        self,
        target: np.ndarray,
        frame: np.ndarray,
        filter_output: FilterOutput,
    ) -> np.ndarray:
        """Copy the original frame onto the target.

        Args:
            target: current composited image of shape (H, W, 3), dtype uint8
            frame: original video frame of shape (H, W, 3), dtype uint8
            filter_output: output from the filter (unused)

        Returns:
            the frame blended onto target according to opacity
        """
        if self.opacity >= 1.0:
            return frame.copy()
        blended = (
            target.astype(np.float32) * (1.0 - self.opacity)
            + frame.astype(np.float32) * self.opacity
        )
        return blended.astype(np.uint8)
