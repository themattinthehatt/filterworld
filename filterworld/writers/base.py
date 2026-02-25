"""Writer base class."""

from abc import ABC, abstractmethod

import numpy as np


class Writer(ABC):
    """Base class for frame writers.

    Subclasses must implement `write_frame` and `close`.
    """

    @abstractmethod
    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single rendered frame.

        Args:
            frame: image array of shape (H, W, 3), dtype uint8, RGB order
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Finalize and release resources."""
        ...

    def __enter__(self) -> 'Writer':
        """Support use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close on context manager exit."""
        self.close()
