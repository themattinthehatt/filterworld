"""Configuration loading and dataclasses."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class LayoutConfig:
    """Layout configuration.

    Args:
        type: layout type, currently only 'grid' is supported
        rows: number of rows in the grid
        cols: number of columns in the grid
    """

    type: str = 'grid'
    rows: int = 1
    cols: int = 1


@dataclass
class PaneConfig:
    """Configuration for a single pane.

    Args:
        layers: list of layer definitions (to be fleshed out later)
        label: optional display label
    """

    layers: list[dict] = field(default_factory=list)
    label: str | None = None


@dataclass
class OutputConfig:
    """Output video configuration.

    Args:
        fps: output fps; null means use the input video's fps
        codec: four-character codec code
        width: output frame width in pixels; null means use input width
        height: output frame height in pixels; null means use input height
    """

    fps: float | None = None
    codec: str = 'mp4v'
    width: int | None = None
    height: int | None = None


@dataclass
class Config:
    """Top-level filterworld configuration.

    Args:
        layout: layout settings
        panes: list of pane configurations
        output: output video settings
    """

    layout: LayoutConfig = field(default_factory=LayoutConfig)
    panes: list[PaneConfig] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str | None) -> Config:
    """Load configuration from a YAML file, or return defaults.

    Args:
        config_path: path to the YAML config file, or None for defaults

    Returns:
        parsed Config instance
    """
    if config_path is None:
        return Config()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'config file not found: {config_path}')

    logger.info('loading config from %s', path)
    raw = yaml.safe_load(path.read_text()) or {}

    layout_raw = raw.get('layout', {})
    layout = LayoutConfig(
        type=layout_raw.get('type', 'grid'),
        rows=layout_raw.get('rows', 1),
        cols=layout_raw.get('cols', 1),
    )

    panes_raw = raw.get('panes', []) or []
    panes = [
        PaneConfig(
            layers=p.get('layers', []),
            label=p.get('label'),
        )
        for p in panes_raw
    ]

    output_raw = raw.get('output', {})
    output = OutputConfig(
        fps=output_raw.get('fps'),
        codec=output_raw.get('codec', 'mp4v'),
        width=output_raw.get('width'),
        height=output_raw.get('height'),
    )

    return Config(layout=layout, panes=panes, output=output)
