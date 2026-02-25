"""DINOv1 ViT feature extraction filter."""

import logging
import math

import numpy as np
import torch
from transformers import ViTImageProcessor, ViTModel

from filterworld.filters.base import FeatureOutput, Filter

logger = logging.getLogger(__name__)


class DINOv1Filter(Filter):
    """Extracts spatial features from frames using a DINOv1 ViT model.

    Args:
        model_name: Hugging Face model identifier (e.g. 'facebook/dino-vits16')
        resolution: input image resolution in pixels; must be divisible by patch_size.
            None uses the model default (typically 224).
    """

    def __init__(self, model_name: str, resolution: int | None = None) -> None:
        self.model_name = model_name
        self.resolution = resolution
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('loading DINOv1 model %s on %s', model_name, self.device)

        processor_kwargs = {}
        if resolution is not None:
            processor_kwargs['size'] = {'height': resolution, 'width': resolution}
            processor_kwargs['crop_size'] = {'height': resolution, 'width': resolution}
        self.processor = ViTImageProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        self.model.eval()
        self.model.to(self.device)

        self._frame_idx = 0

    def process_frame(self, frame: np.ndarray) -> FeatureOutput:
        """Extract spatial features from a single video frame.

        Args:
            frame: numpy array of shape (H, W, 3) in RGB order, dtype uint8

        Returns:
            FeatureOutput with features of shape (D, H_feat, W_feat)
        """
        inputs = self.processor(images=frame, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values, interpolate_pos_encoding=True)

        # drop CLS token, reshape to spatial grid
        hidden = outputs.last_hidden_state[:, 1:, :]  # (1, S, D)
        s = hidden.shape[1]
        d = hidden.shape[2]
        h = w = int(math.isqrt(s))

        features = hidden[0].reshape(h, w, d).permute(2, 0, 1)  # (D, H, W)
        features_np = features.cpu().numpy()

        idx = self._frame_idx
        self._frame_idx += 1

        return FeatureOutput(frame_idx=idx, features=features_np)
