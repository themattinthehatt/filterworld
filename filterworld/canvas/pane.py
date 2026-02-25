"""Pane: one panel in the final frame."""

import numpy as np

from filterworld.filters.base import FilterOutput
from filterworld.layers.base import Layer


class Pane:
    """A single panel that composites a stack of layers.

    A Pane holds an ordered list of Layers. When rendered, it creates a
    blank target image and applies each layer in order, producing a
    single composited image.

    Args:
        layers: ordered list of layers to composite, bottom to top
        label: optional display label for this pane
    """

    def __init__(
        self,
        layers: list[Layer] | None = None,
        label: str | None = None,
    ) -> None:
        self.layers = layers or []
        self.label = label

    def add_layer(self, layer: Layer) -> None:
        """Append a layer to the top of the stack.

        Args:
            layer: layer to add
        """
        self.layers.append(layer)

    def render(
        self,
        frame: np.ndarray,
        filter_output: FilterOutput,
    ) -> np.ndarray:
        """Composite all layers into a single image.

        Args:
            frame: original video frame of shape (H, W, 3), dtype uint8
            filter_output: output from the filter for this frame

        Returns:
            composited image of shape (H, W, 3), dtype uint8
        """
        h, w = frame.shape[:2]
        target = np.zeros((h, w, 3), dtype=np.uint8)
        for layer in self.layers:
            target = layer.render(target, frame, filter_output)
        return target
