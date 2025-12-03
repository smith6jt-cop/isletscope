"""Radial distance analysis around islets.

This module implements functions to compute radial distance maps and quantitate
cell distributions relative to the centroid of detected islets.  The core
class, :class:`RadialAnalyzer`, accepts the binary mask of islets and
optionally other cell masks (e.g. CD3+ cells) and returns radial bin
statistics.  The radial bins extend both inside and outside the islet
boundary at regular intervals measured in pixels.  Overlap maps between
nearby islets can also be derived to approximate lobular organization.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class RadialAnalyzer:
    """Compute radial distance maps and statistics around islets.

    For each islet, the analyzer computes the Euclidean distance from every
    pixel to the islet centroid (or boundary) and bins those distances into
    shells of equal width.  Distances inside the islet are negative and
    distances outside are positive.
    """

    def __init__(
        self,
        bin_size: int = 20,
        max_radius: Optional[int] = None,
        distance_strategy: str = "centroid",
    ) -> None:
        """Initialize the analyzer.

        Args:
            bin_size: Width of each radial bin in pixels.  A smaller value
                produces finer shells.
            max_radius: Optional maximum radius to consider.  If ``None``,
                distances up to the image diagonal are used.
            distance_strategy: ``'centroid'`` (default) measures distance from
                the centroid; ``'boundary'`` measures distance from the islet
                boundary using a distance transform to better handle lobulated
                shapes.
        """
        self.bin_size = bin_size
        self.max_radius = max_radius
        self.distance_strategy = distance_strategy

    def compute_distance_map(self, islet_mask: np.ndarray, strategy: Optional[str] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Compute the signed distance map from the islet centroid or boundary."""
        strat = (strategy or self.distance_strategy).lower()
        ys, xs = np.nonzero(islet_mask)
        if len(xs) == 0:
            raise ValueError("Islet mask is empty; cannot compute centroid.")
        centroid_row = int(np.mean(ys))
        centroid_col = int(np.mean(xs))
        H, W = islet_mask.shape

        if strat == "boundary":
            inside = cv2.distanceTransform(islet_mask.astype(np.uint8), cv2.DIST_L2, 3)
            outside = cv2.distanceTransform((1 - islet_mask).astype(np.uint8), cv2.DIST_L2, 3)
            signed_dist = outside
            signed_dist[islet_mask > 0] = -inside[islet_mask > 0]
            return signed_dist, (centroid_row, centroid_col)

        Y, X = np.indices((H, W))
        dist = np.sqrt((Y - centroid_row) ** 2 + (X - centroid_col) ** 2)
        signed_dist = dist.copy()
        signed_dist[islet_mask > 0] *= -1
        return signed_dist, (centroid_row, centroid_col)

    def radial_bins(self, dist_map: np.ndarray) -> np.ndarray:
        """Assign each pixel to a radial bin based on its signed distance."""
        bins = np.zeros_like(dist_map, dtype=np.int32)
        max_dist = np.max(dist_map)
        if self.max_radius is not None:
            max_dist = min(max_dist, self.max_radius)
        pos_mask = dist_map > 0
        bins[pos_mask] = (dist_map[pos_mask] / self.bin_size).astype(int)
        neg_mask = dist_map < 0
        bins[neg_mask] = -((np.abs(dist_map[neg_mask]) / self.bin_size).astype(int) + 1)
        return bins

    def summarize(self, bins: np.ndarray, cell_mask: Optional[np.ndarray] = None) -> Dict[int, int]:
        """Compute cell counts per radial bin."""
        if cell_mask is not None:
            idxs = np.nonzero(cell_mask)
            bin_vals = bins[idxs]
        else:
            bin_vals = bins.flatten()
        unique_bins, counts = np.unique(bin_vals, return_counts=True)
        return {int(b): int(c) for b, c in zip(unique_bins, counts)}

    def analyze_islets(
        self,
        islet_labels: np.ndarray,
        cell_masks: Optional[Dict[str, np.ndarray]] = None,
        strategy: Optional[str] = None,
    ) -> Dict[int, Dict[str, object]]:
        """Compute per‑islet radial summaries for one or more cell populations."""
        if cell_masks is None:
            cell_masks = {}
        results: Dict[int, Dict[str, object]] = {}
        for islet_id in np.unique(islet_labels):
            if islet_id == 0:
                continue
            mask = islet_labels == islet_id
            if not mask.any():
                continue
            dist_map, centroid = self.compute_distance_map(mask, strategy=strategy)
            bins = self.radial_bins(dist_map)
            per_cell = {name: self.summarize(bins, cm) for name, cm in cell_masks.items()}
            results[int(islet_id)] = {
                "centroid": centroid,
                "area": int(mask.sum()),
                "bin_edges_px": self.bin_size,
                "bins": per_cell if per_cell else self.summarize(bins),
            }
        return results

    def shell_overlap_map(self, islet_labels: np.ndarray, strategy: Optional[str] = None) -> np.ndarray:
        """Return a map counting how many islet shells overlap each pixel."""
        overlap = np.zeros_like(islet_labels, dtype=np.int32)
        for islet_id in np.unique(islet_labels):
            if islet_id == 0:
                continue
            mask = islet_labels == islet_id
            dist_map, _ = self.compute_distance_map(mask, strategy=strategy)
            bins = self.radial_bins(dist_map)
            # Only consider shells outside the islet boundary for overlap assessment
            overlap += (bins >= 0).astype(np.int32)
        return overlap
