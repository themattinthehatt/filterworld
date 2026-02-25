"""Renders CNN/ViT feature maps."""

import cv2
import numpy as np

from filterworld.filters.base import FeatureOutput, FilterOutput
from filterworld.layers.base import Layer


class FeatureLayer(Layer):
    """Layer that visualizes spatial feature maps from a model.

    Reduces a high-dimensional feature tensor to a 3-channel image
    for display.

    Args:
        method: reduction method to convert features to RGB.
            'first3' takes the first 3 channels.
        opacity: layer opacity in [0.0, 1.0]
    """

    def __init__(self, method: str = 'first3', opacity: float = 1.0) -> None:
        super().__init__(opacity=opacity)
        self.method = method

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
        vis_resized = cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)

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
