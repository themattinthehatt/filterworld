"""Tests for filterworld.filters.identity_filter."""

import numpy as np

from filterworld.filters.base import FilterOutput
from filterworld.filters.identity_filter import IdentityFilter


class TestIdentityFilter:
    """Test the class IdentityFilter."""

    def test_identity_filter_returns_filter_output(self, sample_frame):
        """IdentityFilter returns a FilterOutput."""
        f = IdentityFilter()
        output = f.process_frame(sample_frame)
        assert isinstance(output, FilterOutput)

    def test_identity_filter_increments_frame_idx(self, sample_frame):
        """IdentityFilter increments frame_idx on each call."""
        f = IdentityFilter()
        out0 = f.process_frame(sample_frame)
        out1 = f.process_frame(sample_frame)
        out2 = f.process_frame(sample_frame)
        assert out0.frame_idx == 0
        assert out1.frame_idx == 1
        assert out2.frame_idx == 2
