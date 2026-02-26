"""Tests for filterworld.pipeline."""

from unittest.mock import patch

import numpy as np
import pytest

from filterworld.filters.base import FilterOutput
from filterworld.filters.identity_filter import IdentityFilter
from filterworld.pipeline import Pipeline, build_filter


class TestBuildFilter:
    """Test the function build_filter."""

    def test_build_filter_identity(self):
        """'identity' returns an IdentityFilter."""
        f = build_filter('identity')
        assert isinstance(f, IdentityFilter)

    def test_build_filter_unknown_raises(self):
        """Unknown model name raises ValueError."""
        with pytest.raises(ValueError, match='unsupported model'):
            build_filter('nonexistent-model')

    def test_build_filter_registry_names(self):
        """All registry names dispatch without error when model loading is mocked."""
        from filterworld.pipeline import _MODEL_REGISTRY

        for name, (hf_name, filter_cls) in _MODEL_REGISTRY.items():
            with patch.object(filter_cls, '__init__', return_value=None):
                f = build_filter(name)
                assert isinstance(f, filter_cls)


class TestPipeline:
    """Test the class Pipeline."""

    def test_pipeline_identity(self, tmp_video_path, tmp_path):
        """End-to-end pipeline with identity filter produces output file."""
        output_path = tmp_path / 'output.mp4'
        pipeline = Pipeline(
            video_path=str(tmp_video_path),
            model_path='identity',
            config_path=None,
            output_path=str(output_path),
        )
        pipeline.run()
        assert output_path.exists()
