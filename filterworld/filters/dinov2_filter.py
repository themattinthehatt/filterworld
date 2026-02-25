"""DINOv2 ViT feature extraction filter."""

import logging

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from filterworld.filters.base import FeatureOutput, Filter

logger = logging.getLogger(__name__)


class DINOv2Filter(Filter):
    """Extracts spatial features from frames using a DINOv2 model.

    Handles CLS and register tokens automatically based on model config.

    Args:
        model_name: Hugging Face model identifier (e.g. 'facebook/dinov2-small')
        resolution: input image resolution in pixels; must be divisible by patch_size.
            None uses the model default (typically 224).
    """

    def __init__(self, model_name: str, resolution: int | None = None) -> None:
        self.model_name = model_name
        self.resolution = resolution
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('loading DINOv2 model %s on %s', model_name, self.device)

        processor_kwargs = {}
        if resolution is not None:
            processor_kwargs['size'] = {'height': resolution, 'width': resolution}
            processor_kwargs['crop_size'] = {'height': resolution, 'width': resolution}
        self.processor = AutoImageProcessor.from_pretrained(model_name, **processor_kwargs)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        self._num_prefix = 1 + getattr(self.model.config, 'num_register_tokens', 0)
        self._patch_size = self.model.config.patch_size
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
            outputs = self.model(
                pixel_values,
                output_hidden_states=False,
            ).last_hidden_state  # (1, S + num_prefix, D)

        # skip CLS + register tokens
        hidden = outputs[:, self._num_prefix:, :]  # (1, S, D)

        # reshape to spatial grid
        _, h_in, w_in = pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3]
        h_feat = h_in // self._patch_size
        w_feat = w_in // self._patch_size

        features = hidden[0].reshape(h_feat, w_feat, -1).permute(2, 0, 1)  # (D, H, W)
        features_np = features.cpu().numpy()

        idx = self._frame_idx
        self._frame_idx += 1

        return FeatureOutput(frame_idx=idx, features=features_np)
