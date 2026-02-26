"""Tests for filterworld.canvas.canvas."""

import numpy as np
import pytest

from filterworld.canvas.canvas import Canvas, GridLayout, _build_layer
from filterworld.config import Config, LayoutConfig, OutputConfig, PaneConfig
from filterworld.filters.base import FilterOutput
from filterworld.layers.feature_layer import FeatureLayer
from filterworld.layers.image_layer import ImageLayer


class TestBuildLayer:
    """Test the function _build_layer."""

    def test_build_image_layer(self):
        """Building an image layer from dict."""
        layer = _build_layer({'type': 'image', 'opacity': 0.5})
        assert isinstance(layer, ImageLayer)

    def test_build_feature_layer(self, sample_pca_path):
        """Building a feature layer from dict."""
        layer = _build_layer({
            'type': 'feature',
            'method': 'pca',
            'pca_path': str(sample_pca_path),
        })
        assert isinstance(layer, FeatureLayer)

    def test_build_unknown_layer_raises(self):
        """Unknown layer type raises ValueError."""
        with pytest.raises(ValueError, match='unknown layer type'):
            _build_layer({'type': 'nonexistent'})


class TestGridLayout:
    """Test the class GridLayout."""

    def test_grid_layout_single_pane(self, sample_frame):
        """Single pane is resized to fill the layout."""
        layout = GridLayout(width=128, height=128, n_cols=1)
        result = layout.arrange([sample_frame])
        assert result.shape == (128, 128, 3)

    def test_grid_layout_multiple_panes(self, sample_frame):
        """Multiple panes are arranged in a grid."""
        layout = GridLayout(width=128, height=64, n_cols=2)
        result = layout.arrange([sample_frame, sample_frame])
        assert result.shape == (64, 128, 3)

    def test_grid_layout_empty(self):
        """Empty panes returns a black frame."""
        layout = GridLayout(width=64, height=64)
        result = layout.arrange([])
        assert result.shape == (64, 64, 3)
        assert np.all(result == 0)


class TestCanvas:
    """Test the class Canvas."""

    def test_canvas_default(self, sample_frame):
        """Default canvas creates a single pane with an image layer."""
        canvas = Canvas(Config())
        assert len(canvas.panes) == 1
        result = canvas.render(sample_frame, FilterOutput())
        assert result.shape == sample_frame.shape

    def test_canvas_from_config(self, sample_frame):
        """Canvas builds from config with multiple panes."""
        config = Config(
            layout=LayoutConfig(type='grid', rows=1, cols=2),
            panes=[
                PaneConfig(layers=[{'type': 'image'}], label='left'),
                PaneConfig(layers=[{'type': 'image'}], label='right'),
            ],
        )
        canvas = Canvas(config)
        assert len(canvas.panes) == 2
        result = canvas.render(sample_frame, FilterOutput())
        assert result.shape[2] == 3
