"""Tests for filterworld.layers.image_layer."""

import numpy as np

from filterworld.filters.base import FilterOutput
from filterworld.layers.image_layer import ImageLayer


class TestImageLayer:
    """Test the class ImageLayer."""

    def test_image_layer_full_opacity(self, sample_frame):
        """Full opacity returns a copy of the frame."""
        layer = ImageLayer(opacity=1.0)
        target = np.zeros_like(sample_frame)
        result = layer.render(target, sample_frame, FilterOutput())
        np.testing.assert_array_equal(result, sample_frame)
        # verify it is a copy, not the same object
        assert result is not sample_frame

    def test_image_layer_partial_opacity(self, sample_frame):
        """Partial opacity blends target and frame."""
        layer = ImageLayer(opacity=0.5)
        target = np.zeros_like(sample_frame)
        result = layer.render(target, sample_frame, FilterOutput())
        expected = (sample_frame.astype(np.float32) * 0.5).astype(np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_image_layer_zero_opacity(self, sample_frame):
        """Zero opacity returns target unchanged."""
        layer = ImageLayer(opacity=0.0)
        target = np.ones_like(sample_frame) * 128
        result = layer.render(target, sample_frame, FilterOutput())
        np.testing.assert_array_equal(result, target)
