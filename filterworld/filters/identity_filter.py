"""Identity filter that passes frames through unchanged."""

import numpy as np

from filterworld.filters.base import Filter, FilterOutput


class IdentityFilter(Filter):
    """A no-op filter for debugging and testing.

    Returns a bare FilterOutput with no model results, allowing the
    pipeline to run end-to-end without a real model.
    """

    def __init__(self) -> None:
        self._frame_idx = 0

    def process_frame(self, frame: np.ndarray) -> FilterOutput:
        """Return an empty FilterOutput for the given frame.

        Args:
            frame: numpy array of shape (H, W, 3) in RGB order, dtype uint8

        Returns:
            a FilterOutput with only the frame index populated
        """
        output = FilterOutput(frame_idx=self._frame_idx)
        self._frame_idx += 1
        return output
