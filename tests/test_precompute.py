"""Tests for filterworld.precompute."""

from unittest.mock import patch

import numpy as np

from filterworld.filters.base import FeatureOutput, Filter
from filterworld.precompute import precompute_pca


class _FakeFilter(Filter):
    """Fake filter that yields FeatureOutput with random features."""

    def __init__(self) -> None:
        self._frame_idx = 0

    def process_frame(self, frame: np.ndarray) -> FeatureOutput:
        """Return a FeatureOutput with random (384, 4, 4) features."""
        features = np.random.randn(384, 4, 4).astype(np.float32)
        output = FeatureOutput(frame_idx=self._frame_idx, features=features)
        self._frame_idx += 1
        return output


class TestPrecomputePca:
    """Test the function precompute_pca."""

    def test_precompute_pca_produces_npz(self, tmp_video_path, tmp_path):
        """precompute_pca saves npz with components and mean of correct shapes."""
        output_path = tmp_path / 'pca.npz'

        with patch(
            'filterworld.precompute.build_filter',
            return_value=_FakeFilter(),
        ):
            precompute_pca(
                video_path=str(tmp_video_path),
                model_path='fake',
                output_path=str(output_path),
                max_frames=5,
            )

        assert output_path.exists()
        data = np.load(output_path)
        assert 'components' in data
        assert 'mean' in data
        assert data['components'].shape == (3, 384)
        assert data['mean'].shape == (384,)
