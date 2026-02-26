"""Tests for filterworld.filters.vitmae_filter."""

import numpy as np
import pytest

from filterworld.filters.base import FeatureOutput


@pytest.mark.slow
class TestViTMAEFilter:
    """Test the class ViTMAEFilter."""

    def test_vitmae_filter_output_shape(self, sample_frame):
        """ViTMAEFilter produces FeatureOutput with correct shape and dtype."""
        from filterworld.filters.vitmae_filter import ViTMAEFilter

        f = ViTMAEFilter('facebook/vit-mae-base')
        output = f.process_frame(sample_frame)

        assert isinstance(output, FeatureOutput)
        assert output.features.ndim == 3
        d, h, w = output.features.shape
        assert d == 768
        assert output.features.dtype == np.float32
