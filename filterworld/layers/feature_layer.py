"""Renders CNN/ViT feature maps."""

import logging

import cv2
import numpy as np

from filterworld.filters.base import FeatureOutput, FilterOutput
from filterworld.layers.base import Layer

logger = logging.getLogger(__name__)


class FeatureLayer(Layer):
    """Layer that visualizes spatial feature maps from a model.

    Reduces a high-dimensional feature tensor to a 3-channel image
    for display.

    Args:
        method: reduction method to convert features to RGB.
            'first3' takes the first 3 channels.
            'pca' projects features using precomputed PCA weights.
        opacity: layer opacity in [0.0, 1.0]
        pca_path: path to .npz file with PCA weights (required when method='pca')
    """

    def __init__(
        self,
        method: str = 'first3',
        opacity: float = 1.0,
        pca_path: str | None = None,
    ) -> None:
        super().__init__(opacity=opacity)
        self.method = method
        self.pca_path = pca_path
        self._pca_components: np.ndarray | None = None
        self._pca_mean: np.ndarray | None = None

        if method == 'pca' and pca_path is None:
            raise ValueError(
                'pca_path is required when method="pca". '
                'run `filterworld precompute` first to generate PCA weights.'
            )

    def render(
        self,
        target: np.ndarray,
        frame: np.ndarray,
        filter_output: FilterOutput,
    ) -> np.ndarray:
        """Render feature map visualization onto the target image.

        If filter_output is not a FeatureOutput, returns target unchanged.

        Args:
            target: current composited image of shape (H, W, 3), dtype uint8
            frame: original video frame of shape (H, W, 3), dtype uint8
            filter_output: output from the filter for this frame

        Returns:
            updated target image of shape (H, W, 3), dtype uint8
        """
        if not isinstance(filter_output, FeatureOutput):
            return target

        features = filter_output.features  # (D, H_feat, W_feat)
        vis = self._reduce(features)  # (H_feat, W_feat, 3), uint8

        h, w = frame.shape[:2]
        vis_resized = cv2.resize(vis, (w, h), interpolation=cv2.INTER_NEAREST)  # cv2.INTER_LINEAR)

        if self.opacity >= 1.0:
            return vis_resized
        blended = (
            target.astype(np.float32) * (1.0 - self.opacity)
            + vis_resized.astype(np.float32) * self.opacity
        )
        return blended.astype(np.uint8)

    def _reduce(self, features: np.ndarray) -> np.ndarray:
        """Reduce feature tensor to a 3-channel uint8 image.

        Args:
            features: feature tensor of shape (D, H, W)

        Returns:
            RGB image of shape (H, W, 3), dtype uint8
        """
        if self.method == 'first3':
            return self._reduce_first3(features)
        if self.method == 'pca':
            return self._reduce_pca(features)
        raise ValueError(f'unsupported feature reduction method: {self.method}')

    def _reduce_first3(self, features: np.ndarray) -> np.ndarray:
        """Take the first 3 channels and normalize each to 0-255.

        Args:
            features: feature tensor of shape (D, H, W) where D >= 3

        Returns:
            RGB image of shape (H, W, 3), dtype uint8
        """
        channels = features[:3]  # (3, H, W)
        result = np.zeros((*channels.shape[1:], 3), dtype=np.uint8)
        for idx_ch in range(3):
            ch = channels[idx_ch]
            ch_min = ch.min()
            ch_max = ch.max()
            if ch_max - ch_min > 0:
                normalized = (ch - ch_min) / (ch_max - ch_min) * 255.0
            else:
                normalized = np.zeros_like(ch)
            result[:, :, idx_ch] = normalized.astype(np.uint8)
        return result

    def _reduce_pca(self, features: np.ndarray) -> np.ndarray:
        """Project features using precomputed PCA weights.

        On first call, loads the PCA weights from disk and caches them.

        Args:
            features: feature tensor of shape (D, H, W)

        Returns:
            RGB image of shape (H, W, 3), dtype uint8
        """
        if self._pca_components is None:
            logger.info('loading PCA weights from %s', self.pca_path)
            data = np.load(self.pca_path)
            self._pca_components = data['components']  # (3, D)
            self._pca_mean = data['mean']  # (D,)

        d, h, w = features.shape
        patches = features.reshape(d, h * w).T  # (H*W, D)
        centered = patches - self._pca_mean  # (H*W, D)
        projected = centered @ self._pca_components.T  # (H*W, 3)

        # normalize to 0-255 using global min/max across all channels
        val_min = projected.min()
        val_max = projected.max()
        if val_max - val_min > 0:
            normalized = (projected - val_min) / (val_max - val_min) * 255.0
        else:
            normalized = np.zeros_like(projected)

        return normalized.astype(np.uint8).reshape(h, w, 3)
