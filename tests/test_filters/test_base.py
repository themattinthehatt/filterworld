"""Tests for filterworld.filters.base."""

import numpy as np

from filterworld.filters.base import FeatureOutput, FilterOutput


class TestFilterOutput:
    """Test the class FilterOutput."""

    def test_filter_output_defaults(self):
        """FilterOutput has correct defaults."""
        output = FilterOutput()
        assert output.frame_idx == 0
        assert output.metadata == {}

    def test_filter_output_custom(self):
        """FilterOutput accepts custom values."""
        output = FilterOutput(frame_idx=5, metadata={'key': 'value'})
        assert output.frame_idx == 5
        assert output.metadata == {'key': 'value'}


class TestFeatureOutput:
    """Test the class FeatureOutput."""

    def test_feature_output_stores_features(self):
        """FeatureOutput stores a features array."""
        features = np.random.randn(384, 8, 8).astype(np.float32)
        output = FeatureOutput(frame_idx=1, features=features)
        assert output.frame_idx == 1
        assert output.features.shape == (384, 8, 8)

    def test_feature_output_default_features(self):
        """FeatureOutput default features is an empty array."""
        output = FeatureOutput()
        assert output.features.shape == (0,)
