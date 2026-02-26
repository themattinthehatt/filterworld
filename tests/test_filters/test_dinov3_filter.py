"""Tests for filterworld.filters.dinov3_filter."""

import numpy as np
import pytest

from filterworld.filters.base import FeatureOutput


def _has_hf_auth():
    """Check whether HuggingFace authentication is available."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
        return True
    except Exception:
        return False


@pytest.mark.slow
class TestDINOv3Filter:
    """Test the class DINOv3Filter."""

    @pytest.mark.skipif(not _has_hf_auth(), reason='no HuggingFace auth')
    def test_dinov3_filter_output_shape(self, sample_frame):
        """DINOv3Filter produces FeatureOutput with correct shape when authed."""
        from filterworld.filters.dinov3_filter import DINOv3Filter

        f = DINOv3Filter('facebook/dinov3-vits16-pretrain-lvd1689m')
        output = f.process_frame(sample_frame)

        assert isinstance(output, FeatureOutput)
        assert output.features.ndim == 3
        d, h, w = output.features.shape
        assert d == 384
        assert output.features.dtype == np.float32

    @pytest.mark.skipif(_has_hf_auth(), reason='has HuggingFace auth')
    def test_dinov3_filter_raises_without_auth(self):
        """DINOv3Filter raises RuntimeError without auth."""
        from filterworld.filters.dinov3_filter import DINOv3Filter

        with pytest.raises(RuntimeError, match='cannot access DINOv3 model'):
            DINOv3Filter('facebook/dinov3-vits16-pretrain-lvd1689m')
