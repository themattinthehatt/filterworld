"""Canvas: assembles Panes into a full frame."""

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

from filterworld.canvas.pane import Pane
from filterworld.config import Config
from filterworld.filters.base import FilterOutput
from filterworld.layers.base import Layer
from filterworld.layers.feature_layer import FeatureLayer
from filterworld.layers.image_layer import ImageLayer

logger = logging.getLogger(__name__)

# maps layer type strings to Layer classes
_LAYER_REGISTRY: dict[str, type[Layer]] = {
    'image': ImageLayer,
    'feature': FeatureLayer,
}


def _build_layer(layer_dict: dict) -> Layer:
    """Instantiate a Layer from a config dictionary.

    Args:
        layer_dict: dictionary with a 'type' key and optional kwargs

    Returns:
        an initialized Layer instance

    Raises:
        ValueError: if the layer type is unknown
    """
    layer_dict = dict(layer_dict)  # shallow copy to avoid mutating config
    layer_type = layer_dict.pop('type', 'image')
    cls = _LAYER_REGISTRY.get(layer_type)
    if cls is None:
        raise ValueError(
            f'unknown layer type: {layer_type!r}. '
            f'available types: {list(_LAYER_REGISTRY.keys())}'
        )
    return cls(**layer_dict)


class Layout(ABC):
    """Base class for arranging panes into a final frame.

    Args:
        width: output frame width in pixels
        height: output frame height in pixels
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    @abstractmethod
    def arrange(
        self,
        rendered_panes: list[np.ndarray],
    ) -> np.ndarray:
        """Arrange rendered pane images into a single output frame.

        Args:
            rendered_panes: list of composited pane images

        Returns:
            final output frame of shape (height, width, 3), dtype uint8
        """
        ...


class GridLayout(Layout):
    """Arranges panes in a grid.

    Panes are placed left-to-right, top-to-bottom. Each cell is sized
    to fit evenly within the output dimensions.

    Args:
        width: output frame width in pixels
        height: output frame height in pixels
        n_cols: number of columns in the grid
    """

    def __init__(self, width: int, height: int, n_cols: int = 2) -> None:
        super().__init__(width, height)
        self.n_cols = n_cols

    def arrange(
        self,
        rendered_panes: list[np.ndarray],
    ) -> np.ndarray:
        """Arrange rendered pane images in a grid.

        Args:
            rendered_panes: list of composited pane images

        Returns:
            final output frame of shape (height, width, 3), dtype uint8
        """
        n_panes = len(rendered_panes)
        if n_panes == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # single pane: just resize to fill
        if n_panes == 1:
            return cv2.resize(rendered_panes[0], (self.width, self.height))

        n_rows = (n_panes + self.n_cols - 1) // self.n_cols
        cell_w = self.width // self.n_cols
        cell_h = self.height // n_rows

        output = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for idx, pane_img in enumerate(rendered_panes):
            row = idx // self.n_cols
            col = idx % self.n_cols
            resized = cv2.resize(pane_img, (cell_w, cell_h))
            y_start = row * cell_h
            x_start = col * cell_w
            output[y_start:y_start + cell_h, x_start:x_start + cell_w] = resized

        return output


class Canvas:
    """Assembles panes into a final output frame.

    Uses configuration to determine layout and pane setup. When no
    panes are configured, creates a default single-pane canvas that
    renders the original frame.

    Args:
        config: parsed Config instance
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self.panes: list[Pane] = []
        self.layout: Layout | None = None

        if config.panes:
            self._build_from_config()
        else:
            self._build_default()

    def _build_default(self) -> None:
        """Build a default single-pane canvas with an image layer."""
        pane = Pane(layers=[ImageLayer()], label='original')
        self.panes = [pane]

    def _build_from_config(self) -> None:
        """Build panes from configuration."""
        for pane_cfg in self._config.panes:
            layers = [_build_layer(layer_dict) for layer_dict in pane_cfg.layers]
            pane = Pane(layers=layers, label=pane_cfg.label)
            self.panes.append(pane)

    def _init_layout(self, frame_width: int, frame_height: int) -> None:
        """Initialize layout from config and frame dimensions.

        Uses config output width/height if set, otherwise falls back
        to the input frame dimensions.

        Args:
            frame_width: input frame width in pixels
            frame_height: input frame height in pixels
        """
        output_cfg = self._config.output
        layout_cfg = self._config.layout

        n_panes = len(self.panes)
        n_cols = layout_cfg.cols
        n_rows = (n_panes + n_cols - 1) // n_cols

        # when the user hasn't set explicit dimensions, scale the canvas
        # so each grid cell matches the input frame size
        w = output_cfg.width or frame_width * n_cols
        h = output_cfg.height or frame_height * n_rows

        if layout_cfg.type == 'grid':
            self.layout = GridLayout(width=w, height=h, n_cols=layout_cfg.cols)
        else:
            raise ValueError(f'unsupported layout type: {layout_cfg.type}')

    def render(
        self,
        frame: np.ndarray,
        filter_output: FilterOutput,
    ) -> np.ndarray:
        """Render all panes and arrange them into a final frame.

        Args:
            frame: original video frame of shape (H, W, 3), dtype uint8
            filter_output: output from the filter for this frame

        Returns:
            final output frame, dtype uint8
        """
        if self.layout is None:
            h, w = frame.shape[:2]
            self._init_layout(w, h)

        rendered_panes = [
            pane.render(frame, filter_output) for pane in self.panes
        ]
        return self.layout.arrange(rendered_panes)
