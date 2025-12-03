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

from typing import Dict, List, Tuple

import cv2
import numpy as np


class IsletDetector:
    """Detect islet‑like clusters from endocrine cell masks.

    This detector takes a binary mask indicating endocrine‑positive cells and
    identifies connected components that meet specified size, density and
    cell‑count criteria as islets.  Smaller clusters or single cells are labeled
    accordingly.  Nearby fragments can be merged to better capture irregular or
    lobulated islets common in diseased tissue.
    """

    def __init__(
        self,
        min_islet_area: int = 500,
        min_cell_count: int = 10,
        min_density: float = 0.2,
        merge_distance: int = 6,
    ) -> None:
        """Initialize the detector.

        Args:
            min_islet_area: Minimum number of pixels for a connected
                component to be considered an islet.
            min_cell_count: Minimum number of cell objects (connected
                component labels from segmentation) required to classify a
                cluster as an islet.
            min_density: Minimum ratio of cluster area to its convex hull area.
                Helps filter sparse false positives.
            merge_distance: Radius (in pixels) used to merge nearby endocrine
                fragments into a single candidate islet.
        """
        self.min_islet_area = min_islet_area
        self.min_cell_count = min_cell_count
        self.min_density = min_density
        self.merge_distance = merge_distance

    def detect(
        self,
        endocrine_mask: np.ndarray,
        cell_labels: np.ndarray,
        cd3_mask: np.ndarray | None = None,
    ) -> Dict[str, np.ndarray]:
        """Detect islets and return labeled masks.

        Args:
            endocrine_mask: Binary mask of endocrine‑positive cells (e.g. insulin+
                or glucagon+), shape ``(H, W)`` with values ``0`` or ``1``.
            cell_labels: Labeled array from cell segmentation where each cell
                has a unique integer identifier.  The background should be
                labeled ``0``.
            cd3_mask: Optional mask of CD3+ cells to quantify peri‑islet T cell
                density.

        Returns:
            Dictionary containing:
                * ``islet_labels`` – array of shape ``(H, W)`` where each
                  islet cluster has a unique positive integer label and
                  non‑islet regions are ``0``.
                * ``islet_mask`` – binary mask marking all pixels belonging to
                  detected islets.
                * ``single_mask`` – binary mask marking endocrine cells not
                  assigned to an islet.
                * ``metrics`` – list of per‑islet dictionaries summarizing area,
                  cell count, centroid, solidity and optional CD3 counts.
        """
        refined_mask = self._merge_fragments(endocrine_mask.astype(np.uint8))
        num_clusters, cluster_labels = cv2.connectedComponents(refined_mask, connectivity=8)
        islet_labels = np.zeros_like(cluster_labels, dtype=np.int32)
        islet_mask = np.zeros_like(cluster_labels, dtype=np.uint8)
        single_mask = np.zeros_like(cluster_labels, dtype=np.uint8)
        metrics: List[Dict[str, float | int | Tuple[int, int]]] = []
        current_islet_id = 1

        for cid in range(1, num_clusters):
            cluster_pixels = cluster_labels == cid
            area = int(cluster_pixels.sum())
            if area == 0:
                continue
            cell_ids = np.unique(cell_labels[cluster_pixels])
            cell_ids = cell_ids[cell_ids > 0]
            cell_count = int(len(cell_ids))
            solidity = self._compute_solidity(cluster_pixels)
            meets_criteria = (
                area >= self.min_islet_area or cell_count >= self.min_cell_count
            ) and solidity >= self.min_density

            if meets_criteria:
                islet_labels[cluster_pixels] = current_islet_id
                islet_mask[cluster_pixels] = 1
                centroid = tuple(int(v) for v in np.mean(np.column_stack(np.nonzero(cluster_pixels)), axis=0))
                cd3_count = int(cd3_mask[cluster_pixels].sum()) if cd3_mask is not None else 0
                metrics.append(
                    {
                        "id": current_islet_id,
                        "area": area,
                        "cell_count": cell_count,
                        "solidity": float(solidity),
                        "centroid": centroid,  # (row, col)
                        "cd3_pixels": cd3_count,
                    }
                )
                current_islet_id += 1
            else:
                single_mask[cluster_pixels] = 1

        return {
            "islet_labels": islet_labels,
            "islet_mask": islet_mask,
            "single_mask": single_mask,
            "metrics": metrics,
        }

    # Internal helpers -----------------------------------------------------

    def _merge_fragments(self, mask: np.ndarray) -> np.ndarray:
        """Merge nearby endocrine fragments to better capture irregular islets."""
        if self.merge_distance <= 0:
            return (mask > 0).astype(np.uint8)
        k = 2 * self.merge_distance + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        return (dilated > 0).astype(np.uint8)

    def _compute_solidity(self, cluster_mask: np.ndarray) -> float:
        """Compute solidity (area / convex hull area) of a cluster."""
        contour_mask = cluster_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        area = float(cv2.contourArea(contours[0]))
        hull = cv2.convexHull(contours[0])
        hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
        if hull_area == 0.0:
            return 0.0
        return area / hull_area
