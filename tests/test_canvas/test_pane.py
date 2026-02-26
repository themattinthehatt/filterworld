"""Tests for filterworld.canvas.pane."""

import numpy as np

from filterworld.canvas.pane import Pane
from filterworld.filters.base import FilterOutput
from filterworld.layers.image_layer import ImageLayer


class TestPane:
    """Test the class Pane."""

    def test_pane_render_with_image_layer(self, sample_frame):
        """Pane renders an ImageLayer correctly."""
        pane = Pane(layers=[ImageLayer()])
        result = pane.render(sample_frame, FilterOutput())
        np.testing.assert_array_equal(result, sample_frame)

    def test_pane_add_layer(self):
        """add_layer appends a layer to the stack."""
        pane = Pane()
        assert len(pane.layers) == 0
        pane.add_layer(ImageLayer())
        assert len(pane.layers) == 1

    def test_pane_empty_layers(self, sample_frame):
        """Pane with no layers returns a black image."""
        pane = Pane()
        result = pane.render(sample_frame, FilterOutput())
        assert result.shape == sample_frame.shape
        assert np.all(result == 0)
