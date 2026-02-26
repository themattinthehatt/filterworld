"""Tests for filterworld.media.video."""

import numpy as np
import pytest

from filterworld.media.video import VideoReader


class TestVideoReader:
    """Test the class VideoReader."""

    def test_video_reader_properties(self, tmp_video_path):
        """VideoReader reports correct properties."""
        reader = VideoReader(str(tmp_video_path))
        assert reader.width == 64
        assert reader.height == 64
        assert reader.frame_count == 5
        assert reader.fps > 0

    def test_video_reader_iteration(self, tmp_video_path):
        """VideoReader yields frames with correct shape and dtype."""
        reader = VideoReader(str(tmp_video_path))
        frames = list(reader)
        assert len(frames) == 5
        for frame in frames:
            assert frame.shape == (64, 64, 3)
            assert frame.dtype == np.uint8

    def test_video_reader_len(self, tmp_video_path):
        """len(reader) returns frame count."""
        reader = VideoReader(str(tmp_video_path))
        assert len(reader) == 5

    def test_video_reader_missing_file(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            VideoReader('/nonexistent/video.mp4')
