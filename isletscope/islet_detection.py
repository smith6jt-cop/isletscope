"""Detect islets and classify endocrine cells.

This module provides algorithms to distinguish pancreatic islet clusters from
isolated endocrine cells in stained tissue sections.  It operates on
segmentation masks produced by :mod:`isletscope.segmentation` and uses
connected component analysis and simple heuristics to separate clusters
(islets) from single cells.

Because islet morphology varies with disease state and preparation, the
thresholds used here are configurable.  In future versions this module
may integrate a machine learning classifier trained on curated islet
annotations.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, List, Tuple


class IsletDetector:
    """Detect islet‑like clusters from endocrine cell masks.

    This detector takes a binary mask indicating endocrine‑positive cells and
    identifies connected components that meet specified size criteria as
    islets.  Smaller clusters or single cells are labeled accordingly.
    """

    def __init__(self, min_islet_area: int = 500, min_cell_count: int = 10) -> None:
        """Initialize the detector.

        Args:
            min_islet_area: Minimum number of pixels for a connected
                component to be considered an islet.
            min_cell_count: Minimum number of cell objects (connected
                component labels from segmentation) required to classify a
                cluster as an islet.  If both criteria are met, the cluster
                is labeled as an islet.
        """
        self.min_islet_area = min_islet_area
        self.min_cell_count = min_cell_count

    def detect(self, endocrine_mask: np.ndarray, cell_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect islets and return labeled masks.

        Args:
            endocrine_mask: Binary mask of endocrine‑positive cells (e.g. insulin+
                or glucagon+), shape ``(H, W)`` with values ``0`` or ``1``.
            cell_labels: Labeled array from cell segmentation where each cell
                has a unique integer identifier.  The background should be
                labeled ``0``.

        Returns:
            Dictionary containing:
                * ``islet_labels`` – array of shape ``(H, W)`` where each
                  islet cluster has a unique positive integer label and
                  non‑islet regions are ``0``.
                * ``islet_mask`` – binary mask marking all pixels belonging to
                  detected islets.
                * ``single_mask`` – binary mask marking endocrine cells not
                  assigned to an islet.
        """
        # Label connected endocrine regions
        endocrine_mask_uint = endocrine_mask.astype(np.uint8)
        num_clusters, cluster_labels = cv2.connectedComponents(endocrine_mask_uint, connectivity=8)
        # Prepare output arrays
        islet_labels = np.zeros_like(cluster_labels, dtype=np.int32)
        islet_mask = np.zeros_like(cluster_labels, dtype=np.uint8)
        single_mask = np.zeros_like(cluster_labels, dtype=np.uint8)
        current_islet_id = 1
        # Iterate over clusters (skip background label 0)
        for cid in range(1, num_clusters):
            cluster_pixels = (cluster_labels == cid)
            area = np.count_nonzero(cluster_pixels)
            # Count unique cell labels within this cluster
            cell_ids = np.unique(cell_labels[cluster_pixels])
            # Remove background label 0
            cell_ids = cell_ids[cell_ids > 0]
            cell_count = len(cell_ids)
            if area >= self.min_islet_area or cell_count >= self.min_cell_count:
                # Classify as islet
                islet_labels[cluster_pixels] = current_islet_id
                islet_mask[cluster_pixels] = 1
                current_islet_id += 1
            else:
                # Single endocrine cells or small clusters
                single_mask[cluster_pixels] = 1
        return {
            'islet_labels': islet_labels,
            'islet_mask': islet_mask,
            'single_mask': single_mask,
        }
