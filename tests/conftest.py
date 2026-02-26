"""Shared test fixtures."""

import cv2
import numpy as np
import pytest
import yaml


@pytest.fixture
def tmp_video_path(tmp_path):
    """Generate a 5-frame 64x64 random-pixel mp4 video, return its path."""
    video_path = tmp_path / 'test_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 64))
    for _ in range(5):
        frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.fixture
def sample_frame():
    """Return a single 64x64 random uint8 RGB numpy array."""
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_features():
    """Return a random (384, 8, 8) float32 array simulating ViT-S patch features."""
    return np.random.randn(384, 8, 8).astype(np.float32)


@pytest.fixture
def sample_pca_path(tmp_path):
    """Save a fake PCA npz with components (3, 384) and mean (384,), return path."""
    pca_path = tmp_path / 'pca.npz'
    np.savez(
        pca_path,
        components=np.random.randn(3, 384).astype(np.float32),
        mean=np.random.randn(384).astype(np.float32),
    )
    return pca_path


@pytest.fixture
def dino_config_path(tmp_path, sample_pca_path):
    """Write a dino.yaml config pointing to the pca fixture, return path."""
    config = {
        'layout': {'type': 'grid', 'rows': 1, 'cols': 2},
        'panes': [
            {'layers': [{'type': 'image'}], 'label': 'original'},
            {
                'layers': [
                    {'type': 'feature', 'method': 'pca', 'pca_path': str(sample_pca_path)},
                ],
                'label': 'features',
            },
        ],
    }
    config_path = tmp_path / 'dino.yaml'
    config_path.write_text(yaml.dump(config))
    return config_path
