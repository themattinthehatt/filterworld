"""Tests for filterworld.filters.dinov2_filter."""

import numpy as np
import pytest

from filterworld.filters.base import FeatureOutput


@pytest.mark.slow
class TestDINOv2Filter:
    """Test the class DINOv2Filter."""

    def test_dinov2_filter_output_shape(self, sample_frame):
        """DINOv2Filter produces FeatureOutput with correct shape and dtype."""
        from filterworld.filters.dinov2_filter import DINOv2Filter

        f = DINOv2Filter('facebook/dinov2-small')
        output = f.process_frame(sample_frame)

        assert isinstance(output, FeatureOutput)
        assert output.features.ndim == 3
        d, h, w = output.features.shape
        assert d == 384
        assert output.features.dtype == np.float32
