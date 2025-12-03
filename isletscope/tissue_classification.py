"""Tissue classification for pancreatic histology images.

This module implements a multi‑class segmentation of pancreatic tissue into
parenchymal acinar regions, connective tissue/space, ducts and vessels,
background glass and a catch‑all ``other`` class.  The classifier operates on
RGB images and uses k‑means clustering in the CIELAB color space with
brightness‑based heuristics to map clusters to semantic labels.  Bright
luminal structures are additionally refined using morphological operations to
separate ducts/vessels from background.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


class TissueClassifier:
    """Classify pancreatic tissue into multiple classes using k‑means."""

    def __init__(
        self,
        n_clusters: int = 4,
        criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        luminal_threshold: int = 220,
        min_lumen_area: int = 64,
    ) -> None:
        """Initialize the classifier.

        Args:
            n_clusters: Number of color clusters to use (``>=4`` recommended).
            criteria: Termination criteria for k‑means clustering.
            luminal_threshold: Intensity threshold (0–255) to identify bright
                lumens for ducts/vessels.
            min_lumen_area: Minimum area (pixels) to keep a lumen region.
        """
        self.n_clusters = n_clusters
        self.criteria = criteria
        self.luminal_threshold = luminal_threshold
        self.min_lumen_area = min_lumen_area

    def classify(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Classify tissue pixels.

        Args:
            image: RGB image of shape ``(H, W, 3)``.

        Returns:
            Dictionary with keys ``'parenchyma'``, ``'connective'``,
            ``'ducts_vessels'``, ``'background'`` and ``'other'``, each
            containing a binary mask of the corresponding tissue class.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        H, W, _ = lab.shape
        lab_flat = lab.reshape(-1, 3).astype(np.float32)
        _, labels, centers = cv2.kmeans(lab_flat, self.n_clusters, None, self.criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        L_values = centers[:, 0]
        sorted_idx = np.argsort(L_values)
        parenchyma_cluster = sorted_idx[0]
        connective_cluster = sorted_idx[-2] if self.n_clusters >= 3 else sorted_idx[-1]
        background_cluster = sorted_idx[-1]

        semantic_masks = {
            "parenchyma": np.zeros((H * W,), dtype=np.uint8),
            "connective": np.zeros((H * W,), dtype=np.uint8),
            "ducts_vessels": np.zeros((H * W,), dtype=np.uint8),
            "background": np.zeros((H * W,), dtype=np.uint8),
            "other": np.zeros((H * W,), dtype=np.uint8),
        }

        semantic_masks["parenchyma"][labels == parenchyma_cluster] = 1
        semantic_masks["connective"][labels == connective_cluster] = 1
        semantic_masks["background"][labels == background_cluster] = 1
        for oc in sorted_idx:
            if oc in {parenchyma_cluster, connective_cluster, background_cluster}:
                continue
            semantic_masks["other"][labels == oc] = 1

        # Refine ducts/vessels using bright lumen detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lumen = (gray > self.luminal_threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        lumen = cv2.morphologyEx(lumen, cv2.MORPH_OPEN, kernel, iterations=1)
        num, lumen_labels, stats, _ = cv2.connectedComponentsWithStats(lumen, connectivity=8)
        refined_lumen = np.zeros_like(lumen)
        for cid in range(1, num):
            if stats[cid, cv2.CC_STAT_AREA] >= self.min_lumen_area:
                refined_lumen[lumen_labels == cid] = 1
        # Assign lumen regions to ducts/vessels and remove from background
        refined_lumen_flat = refined_lumen.flatten()
        semantic_masks["ducts_vessels"][refined_lumen_flat == 1] = 1
        semantic_masks["background"][refined_lumen_flat == 1] = 0

        # Reshape masks back to image shape
        for key in semantic_masks:
            semantic_masks[key] = semantic_masks[key].reshape(H, W)
        return semantic_masks
