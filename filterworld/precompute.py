"""Precompute PCA weights from video features."""

import logging

import numpy as np
from sklearn.decomposition import PCA

from filterworld.media.video import VideoReader
from filterworld.pipeline import build_filter

logger = logging.getLogger(__name__)


def precompute_pca(
    video_path: str,
    model_path: str,
    output_path: str,
    max_frames: int = 200,
    resolution: int | None = None,
) -> None:
    """Fit PCA on patch embeddings from a video and save weights to disk.

    Runs the specified model on a subset of video frames, collects all
    spatial patch embeddings, fits a 3-component PCA, and saves the
    components and mean to a .npz file.

    Args:
        video_path: path to the input video file
        model_path: model name (e.g. 'dinov2-small')
        output_path: output .npz path for PCA weights
        max_frames: maximum number of frames to use for PCA fitting
        resolution: optional model input resolution override in pixels
    """
    reader = VideoReader(video_path)
    vid_filter = build_filter(model_path, resolution=resolution)
    frame_count = reader.frame_count

    # determine which frame indices to process
    if frame_count <= max_frames:
        indices_selected = set(range(frame_count))
    else:
        indices_selected = set(
            np.random.choice(frame_count, max_frames, replace=False).tolist()
        )

    logger.info(
        'precomputing PCA: %d/%d frames from %s',
        len(indices_selected), frame_count, video_path,
    )

    # collect patch embeddings from selected frames
    embeddings = []
    for idx_frame, frame in enumerate(reader):
        if idx_frame not in indices_selected:
            continue
        filter_output = vid_filter.process_frame(frame)
        features = filter_output.features  # (D, H, W)
        d, h, w = features.shape
        patches = features.reshape(d, h * w).T  # (H*W, D)
        embeddings.append(patches)

        if (len(embeddings) % 50) == 0:
            logger.info('processed %d/%d frames', len(embeddings), len(indices_selected))

    all_embeddings = np.concatenate(embeddings, axis=0)  # (N_total, D)
    logger.info('fitting PCA on %d patch embeddings of dimension %d', *all_embeddings.shape)

    pca = PCA(n_components=3)
    pca.fit(all_embeddings)

    np.savez(
        output_path,
        components=pca.components_,  # (3, D)
        mean=pca.mean_,  # (D,)
    )
    logger.info('saved PCA weights to %s', output_path)
