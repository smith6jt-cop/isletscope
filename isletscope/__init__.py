"""Top level package for IsletScope.

This package provides a unified API for stain normalization, segmentation, islet detection,
radial analysis, tissue classification and 3D anatomical inference.  Each submodule exposes
classes and functions documented in the API reference.  Import the relevant classes as
needed:

    from isletscope.stain import StainNormalizer
    from isletscope.segmentation import CellSegmenter
    from isletscope.islet_detection import IsletDetector
    from isletscope.radial_analysis import RadialAnalyzer
    from isletscope.tissue_classification import TissueClassifier
    from isletscope.spatial_inference import SpatialInferer

See the documentation for usage examples.
"""

from .stain import StainNormalizer
from .segmentation import CellSegmenter
from .islet_detection import IsletDetector
from .radial_analysis import RadialAnalyzer
from .tissue_classification import TissueClassifier
from .spatial_inference import SpatialInferer

__all__ = [
    "StainNormalizer",
    "CellSegmenter",
    "IsletDetector",
    "RadialAnalyzer",
    "TissueClassifier",
    "SpatialInferer",
]
