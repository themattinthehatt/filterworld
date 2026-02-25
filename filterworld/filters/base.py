"""Filter base class and FilterOutput dataclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FilterOutput:
    """Base container for filter results.

    Args:
        frame_idx: index of the frame that produced this output
        metadata: arbitrary key-value pairs for extra information
    """

    frame_idx: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class BBoxOutput(FilterOutput):
    """Bounding box detection results."""

    pass


@dataclass
class SegmentationOutput(FilterOutput):
    """Segmentation mask results."""

    pass


@dataclass
class FeatureOutput(FilterOutput):
    """Feature extraction results."""

    pass


@dataclass
class KeypointOutput(FilterOutput):
    """Keypoint detection results."""

    pass


@dataclass
class DepthOutput(FilterOutput):
    """Depth estimation results."""

    pass


class Filter(ABC):
    """Base class for all filters.

    Subclasses must implement `process_frame` to transform a video frame
    into a `FilterOutput`.
    """

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> FilterOutput:
        """Process a single video frame.

        Args:
            frame: numpy array of shape (H, W, 3) in RGB order, dtype uint8

        Returns:
            filter output containing model results for this frame
        """
        ...
