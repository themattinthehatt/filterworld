"""Wraps Hugging Face transformers models."""

from filterworld.filters.base import Filter


class HuggingFaceFilter(Filter):
    pass


class HFDetectionFilter(HuggingFaceFilter):
    pass


class HFSegmentationFilter(HuggingFaceFilter):
    pass


class HFDepthFilter(HuggingFaceFilter):
    pass


class HFFeatureFilter(HuggingFaceFilter):
    pass
