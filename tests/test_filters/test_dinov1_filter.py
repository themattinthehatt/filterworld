"""Tests for filterworld.filters.dinov1_filter."""

import numpy as np
import pytest

from filterworld.filters.base import FeatureOutput


@pytest.mark.slow
class TestDINOv1Filter:
    """Test the class DINOv1Filter."""

    def test_dinov1_filter_output_shape(self, sample_frame):
        """DINOv1Filter produces FeatureOutput with correct shape and dtype."""
        from filterworld.filters.dinov1_filter import DINOv1Filter

        f = DINOv1Filter('facebook/dino-vits16')
        output = f.process_frame(sample_frame)

        assert isinstance(output, FeatureOutput)
        assert output.features.ndim == 3
        d, h, w = output.features.shape
        assert d == 384
        assert output.features.dtype == np.float32
