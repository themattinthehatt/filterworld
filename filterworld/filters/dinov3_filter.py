"""DINOv3 ViT feature extraction filter."""

import logging

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from filterworld.filters.base import FeatureOutput, Filter

logger = logging.getLogger(__name__)

_DINOV3_ACCESS_HELP = """
================================================================================
DINOv3 Model Access Required
================================================================================

The DINOv3 models are gated on HuggingFace and require authentication.
Please follow these steps to gain access:

1. CREATE A HUGGINGFACE ACCOUNT (if you don't have one):
   - Go to https://huggingface.co/join
   - Sign up with your email

2. REQUEST ACCESS TO THE MODEL:
   - Visit the model page on HuggingFace (e.g.
     https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
   - Click "Agree and access repository"
   - Accept the terms of use
   - You should receive immediate access

3. CREATE AN ACCESS TOKEN:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Give it a name (e.g. "filterworld-dinov3")
   - Select "Read" permission (sufficient for downloading models)
   - Click "Generate token"
   - COPY THE TOKEN - you won't be able to see it again!

4. LOGIN FROM THE TERMINAL:
   pip install -U transformers huggingface_hub
   hf auth login

   When prompted, paste your token and press Enter.

5. RE-RUN YOUR COMMAND

For more information, visit: https://huggingface.co/docs/hub/security-tokens
================================================================================
"""


class DINOv3Filter(Filter):
    """Extracts spatial features from frames using a DINOv3 model.

    DINOv3 models are gated on HuggingFace and require authentication.
    Handles CLS and register tokens automatically based on model config.

    Args:
        model_name: Hugging Face model identifier
            (e.g. 'facebook/dinov3-vits16-pretrain-lvd1689m')
        resolution: input image resolution in pixels; must be divisible by patch_size.
            None uses the model default (typically 224).
    """

    def __init__(self, model_name: str, resolution: int | None = None) -> None:
        self.model_name = model_name
        self.resolution = resolution
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('loading DINOv3 model %s on %s', model_name, self.device)

        processor_kwargs = {}
        if resolution is not None:
            processor_kwargs['size'] = {'height': resolution, 'width': resolution}
            processor_kwargs['crop_size'] = {'height': resolution, 'width': resolution}

        self.processor, self.model = _load_dinov3(model_name, processor_kwargs)
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


def _load_dinov3(
    model_name: str,
    processor_kwargs: dict,
) -> tuple[AutoImageProcessor, AutoModel]:
    """Load DINOv3 model and processor with auth error handling.

    Args:
        model_name: Hugging Face model identifier
        processor_kwargs: extra kwargs for AutoImageProcessor.from_pretrained

    Returns:
        tuple of (processor, model)

    Raises:
        RuntimeError: if model access is denied due to gating
    """
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, **processor_kwargs)
        model = AutoModel.from_pretrained(model_name)
        return processor, model
    except OSError as e:
        if 'gated repo' in str(e).lower():
            print(_DINOV3_ACCESS_HELP)
            raise RuntimeError(
                'cannot access DINOv3 model. '
                'please follow the instructions above to authenticate with HuggingFace.'
            ) from e
        raise
