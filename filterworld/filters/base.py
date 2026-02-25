"""Filter base class and FilterOutput dataclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FilterOutput:
    pass


@dataclass
class BBoxOutput(FilterOutput):
    pass


@dataclass
class SegmentationOutput(FilterOutput):
    pass


@dataclass
class FeatureOutput(FilterOutput):
    pass


@dataclass
class KeypointOutput(FilterOutput):
    pass


@dataclass
class DepthOutput(FilterOutput):
    pass


class Filter(ABC):
    pass
