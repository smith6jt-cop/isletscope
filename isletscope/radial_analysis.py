"""Radial distance analysis around islets.

This module implements functions to compute radial distance maps and quantitate
cell distributions relative to the centroid of detected islets.  The core
class, :class:`RadialAnalyzer`, accepts the binary mask of islets and
optionally other cell masks (e.g. CD3+ cells) and returns radial bin
statistics.  The radial bins extend both inside and outside the islet
boundary at regular intervals measured in pixels.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple


class RadialAnalyzer:
    """Compute radial distance maps and statistics around islets.

    For each islet, the analyzer computes the Euclidean distance from every
    pixel to the islet centroid.  Distances inside the islet are negative and
    distances outside are positive.  The distances are then binned into
    shells of equal width (in pixels).
    """

    def __init__(self, bin_size: int = 20, max_radius: Optional[int] = None) -> None:
        """Initialize the analyzer.

        Args:
            bin_size: Width of each radial bin in pixels.  A smaller value
                produces finer shells.
            max_radius: Optional maximum radius to consider.  If ``None``,
                distances up to the image diagonal are used.
        """
        self.bin_size = bin_size
        self.max_radius = max_radius

    def compute_distance_map(self, islet_mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Compute the signed distance map from the islet centroid.

        The centroid is computed using the mean coordinates of all positive
        pixels in ``islet_mask``.  Distances to the centroid are positive
        outside the islet and negative inside.

        Args:
            islet_mask: Binary mask of an individual islet, shape ``(H, W)``.

        Returns:
            A tuple ``(dist_map, centroid)`` where ``dist_map`` is a float
            array of shape ``(H, W)`` containing signed distances and
            ``centroid`` is the ``(row, col)`` coordinate of the islet centroid.
        """
        ys, xs = np.nonzero(islet_mask)
        if len(xs) == 0:
            raise ValueError("Islet mask is empty; cannot compute centroid.")
        centroid_row = int(np.mean(ys))
        centroid_col = int(np.mean(xs))
        # Create coordinate grid
        H, W = islet_mask.shape
        Y, X = np.indices((H, W))
        # Compute Euclidean distance to centroid
        dist = np.sqrt((Y - centroid_row) ** 2 + (X - centroid_col) ** 2)
        # Negative distances inside islet
        signed_dist = dist.copy()
        signed_dist[islet_mask > 0] *= -1
        return signed_dist, (centroid_row, centroid_col)

    def radial_bins(self, dist_map: np.ndarray) -> np.ndarray:
        """Assign each pixel to a radial bin based on its signed distance.

        Positive distances (outside islet) start at bin index 0 and increase
        outward.  Negative distances (inside islet) start at bin index ``-1``
        for the immediate shell inside the boundary, ``-2`` for the next
        shell towards the centroid, and so on.

        Args:
            dist_map: Signed distance map as returned by
                :meth:`compute_distance_map`.

        Returns:
            Integer array of the same shape as ``dist_map`` where each pixel
            contains the bin index.  Background or empty distances may be
            assigned a bin outside the specified range if ``max_radius`` is
            not set.
        """
        bins = np.zeros_like(dist_map, dtype=np.int32)
        # Determine maximum positive distance
        max_dist = np.max(dist_map)
        if self.max_radius is not None:
            max_dist = min(max_dist, self.max_radius)
        # Positive distances (outside)
        pos_mask = dist_map > 0
        bins[pos_mask] = (dist_map[pos_mask] / self.bin_size).astype(int)
        # Negative distances (inside) – shift by one so that the shell
        # immediately inside the boundary is -1
        neg_mask = dist_map < 0
        bins[neg_mask] = -((np.abs(dist_map[neg_mask]) / self.bin_size).astype(int) + 1)
        return bins

    def summarize(self, bins: np.ndarray, cell_mask: Optional[np.ndarray] = None) -> Dict[int, int]:
        """Compute cell counts per radial bin.

        Args:
            bins: Integer bin assignments for each pixel.
            cell_mask: Optional binary mask of a particular cell type (e.g. CD3+).
                If provided, the histogram is computed only over these pixels.

        Returns:
            Dictionary mapping bin index to the number of cell pixels in that bin.
        """
        if cell_mask is not None:
            idxs = np.nonzero(cell_mask)
            bin_vals = bins[idxs]
        else:
            bin_vals = bins.flatten()
        # Compute histogram using numpy unique
        unique_bins, counts = np.unique(bin_vals, return_counts=True)
        return {int(b): int(c) for b, c in zip(unique_bins, counts)}
