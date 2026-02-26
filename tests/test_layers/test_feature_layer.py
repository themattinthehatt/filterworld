"""Tests for filterworld.layers.feature_layer."""

import numpy as np
import pytest

from filterworld.filters.base import FeatureOutput, FilterOutput
from filterworld.layers.feature_layer import FeatureLayer


class TestFeatureLayer:
    """Test the class FeatureLayer."""

    def test_reduce_first3_produces_uint8_rgb(self, sample_features):
        """_reduce_first3 produces a uint8 (H, W, 3) image."""
        layer = FeatureLayer(method='first3')
        result = layer._reduce_first3(sample_features)
        assert result.dtype == np.uint8
        assert result.shape == (8, 8, 3)

    def test_reduce_pca_produces_correct_shape(self, sample_features, sample_pca_path):
        """_reduce_pca loads npz and produces correct shape."""
        layer = FeatureLayer(method='pca', pca_path=str(sample_pca_path))
        result = layer._reduce_pca(sample_features)
        assert result.dtype == np.uint8
        assert result.shape == (8, 8, 3)

    def test_pca_without_path_raises(self):
        """method='pca' without pca_path raises ValueError."""
        with pytest.raises(ValueError, match='pca_path is required'):
            FeatureLayer(method='pca')

    def test_render_with_non_feature_output_returns_target(self, sample_frame):
        """Rendering with a non-FeatureOutput returns target unchanged."""
        layer = FeatureLayer(method='first3')
        target = np.zeros_like(sample_frame)
        result = layer.render(target, sample_frame, FilterOutput())
        np.testing.assert_array_equal(result, target)

    def test_render_with_feature_output(self, sample_frame, sample_features):
        """Rendering with a FeatureOutput produces a valid image."""
        layer = FeatureLayer(method='first3', opacity=1.0)
        target = np.zeros_like(sample_frame)
        feature_output = FeatureOutput(frame_idx=0, features=sample_features)
        result = layer.render(target, sample_frame, feature_output)
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8
