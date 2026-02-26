"""Tests for filterworld.config."""

import yaml
import pytest

from filterworld.config import Config, LayoutConfig, OutputConfig, PaneConfig, load_config


class TestLoadConfig:
    """Test the function load_config."""

    def test_load_config_default(self):
        """None config_path returns a default Config."""
        config = load_config(None)
        assert isinstance(config, Config)
        assert config.layout.type == 'grid'
        assert config.panes == []

    def test_load_config_from_yaml(self, tmp_path):
        """Loading from a valid yaml file parses correctly."""
        config_data = {
            'layout': {'type': 'grid', 'rows': 2, 'cols': 3},
            'panes': [
                {'layers': [{'type': 'image'}], 'label': 'pane1'},
            ],
            'output': {'fps': 24.0, 'codec': 'avc1'},
        }
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(yaml.dump(config_data))

        config = load_config(str(config_path))
        assert config.layout.rows == 2
        assert config.layout.cols == 3
        assert len(config.panes) == 1
        assert config.panes[0].label == 'pane1'
        assert config.output.fps == 24.0
        assert config.output.codec == 'avc1'

    def test_load_config_missing_file(self):
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config('/nonexistent/path/config.yaml')


class TestDataclasses:
    """Test the config dataclasses."""

    def test_layout_config_defaults(self):
        """LayoutConfig has sensible defaults."""
        layout = LayoutConfig()
        assert layout.type == 'grid'
        assert layout.rows == 1
        assert layout.cols == 1

    def test_output_config_defaults(self):
        """OutputConfig has sensible defaults."""
        output = OutputConfig()
        assert output.fps is None
        assert output.codec == 'mp4v'
        assert output.width is None
        assert output.height is None

    def test_pane_config_defaults(self):
        """PaneConfig has sensible defaults."""
        pane = PaneConfig()
        assert pane.layers == []
        assert pane.label is None
