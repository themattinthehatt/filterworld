"""Canvas: assembles Panes into a full frame."""

from abc import ABC, abstractmethod

import cv2
import numpy as np

from filterworld.canvas.pane import Pane
from filterworld.filters.base import FilterOutput
from filterworld.layers.image_layer import ImageLayer


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

    When no config is provided, creates a default single-pane canvas
    that renders the original frame (identity passthrough).

    Args:
        config_path: optional path to a YAML config file defining
            panes and layout; if None, uses a single-pane default
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.panes: list[Pane] = []
        self.layout: Layout | None = None
        self._config_path = config_path

        if config_path is not None:
            self._load_config(config_path)
        else:
            self._build_default()

    def _build_default(self) -> None:
        """Build a default single-pane canvas with an image layer."""
        pane = Pane(layers=[ImageLayer()], label='original')
        self.panes = [pane]
        # layout will be initialized on first render when we know frame size

    def _load_config(self, config_path: str) -> None:
        """Load canvas configuration from a YAML file.

        Args:
            config_path: path to the YAML config file

        Raises:
            NotImplementedError: config loading is not yet implemented
        """
        raise NotImplementedError('YAML config loading is not yet implemented')

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
        h, w = frame.shape[:2]

        # lazily initialize layout from first frame dimensions
        if self.layout is None:
            n_cols = 1 if len(self.panes) == 1 else 2
            self.layout = GridLayout(width=w, height=h, n_cols=n_cols)

        rendered_panes = [
            pane.render(frame, filter_output) for pane in self.panes
        ]
        return self.layout.arrange(rendered_panes)
