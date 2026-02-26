"""Tests for filterworld.writers.video_writer."""

import numpy as np

from filterworld.writers.video_writer import VideoWriter


class TestVideoWriter:
    """Test the class VideoWriter."""

    def test_video_writer_write_frames(self, tmp_path):
        """VideoWriter writes frames and creates the output file."""
        output_path = tmp_path / 'output.mp4'
        writer = VideoWriter(str(output_path), fps=30.0)
        for _ in range(3):
            frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert output_path.exists()
        assert writer.frame_count == 3

    def test_video_writer_close_idempotent(self, tmp_path):
        """Calling close twice does not raise."""
        output_path = tmp_path / 'output.mp4'
        writer = VideoWriter(str(output_path), fps=30.0)
        frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        writer.write_frame(frame)
        writer.close()
        writer.close()  # should not raise
