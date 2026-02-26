"""Tests for filterworld.cli.main."""

from pathlib import Path

import pytest

from filterworld.cli.main import _derive_output_path, main, parse_args


class TestParseArgs:
    """Test the function parse_args."""

    def test_parse_args_run(self):
        """Run subcommand parses correctly."""
        args = parse_args(['run', 'video.mp4', 'identity'])
        assert args.command == 'run'
        assert args.video == 'video.mp4'
        assert args.model == 'identity'
        assert args.config is None
        assert args.output is None

    def test_parse_args_precompute(self):
        """Precompute subcommand parses correctly."""
        args = parse_args(['precompute', 'video.mp4', 'dinov2-small', '-o', 'pca.npz'])
        assert args.command == 'precompute'
        assert args.video == 'video.mp4'
        assert args.model == 'dinov2-small'
        assert args.output == 'pca.npz'

    def test_parse_args_no_command(self):
        """No subcommand raises SystemExit."""
        with pytest.raises(SystemExit):
            parse_args([])


class TestDeriveOutputPath:
    """Test the function _derive_output_path."""

    def test_derive_output_path_suffix(self):
        """Derives path with _filtered suffix."""
        result = _derive_output_path('video.mp4')
        assert result == 'video_filtered.mp4'

    def test_derive_output_path_with_directory(self):
        """Preserves directory in output path."""
        result = _derive_output_path('/path/to/video.mp4')
        assert result == '/path/to/video_filtered.mp4'


class TestMain:
    """Test the function main."""

    def test_main_run_identity(self, tmp_video_path, tmp_path):
        """End-to-end run with identity filter via main()."""
        output_path = tmp_path / 'output.mp4'
        main(['run', str(tmp_video_path), 'identity', '-o', str(output_path)])
        assert output_path.exists()
