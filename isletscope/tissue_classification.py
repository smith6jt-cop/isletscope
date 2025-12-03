"""Tissue classification for pancreatic histology images.

This module implements a basic multi‑class segmentation of pancreatic tissue
into parenchymal acinar regions, connective tissue/space, and other
structures such as ducts, vessels and nerves.  The classifier operates on
RGB images and uses k‑means clustering in the CIELAB color space to assign
each pixel to one of ``n_clusters`` classes.  Heuristics are applied to map
clusters to semantic tissue classes based on intensity ordering.

For high‑quality segmentation, users are encouraged to train a deep neural
network on annotated data; the present implementation serves as a
placeholder and starting point.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, Tuple


class TissueClassifier:
    """Classify pancreatic tissue into multiple classes using k‑means.

    The classifier clusters pixel colors in LAB space and assigns cluster
    labels to semantic tissue classes based on brightness ordering.  This
    simple method works best when the image is roughly color balanced and
    pre‑normalized using :mod:`isletscope.stain`.
    """

    def __init__(self, n_clusters: int = 3, criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)) -> None:
        """Initialize the classifier.

        Args:
            n_clusters: Number of color clusters to use.  A value of 3
                typically corresponds to parenchyma, connective tissue and
                other structures.
            criteria: Termination criteria for k‑means clustering.
        """
        self.n_clusters = n_clusters
        self.criteria = criteria

    def classify(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Classify tissue pixels.

        Args:
            image: RGB image of shape ``(H, W, 3)``.

        Returns:
            Dictionary with keys ``'parenchyma'``, ``'connective'`` and
            ``'other'``, each containing a binary mask of the corresponding
            tissue class.  The mask arrays have shape ``(H, W)`` and values
            ``0`` or ``1``.
        """
        # Convert to LAB color space for better perceptual clustering
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        H, W, _ = lab.shape
        # Flatten and convert to float32 for k‑means
        lab_flat = lab.reshape(-1, 3).astype(np.float32)
        # Run k‑means
        _, labels, centers = cv2.kmeans(lab_flat, self.n_clusters, None, self.criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        # Determine ordering of clusters by L channel (brightness)
        # Higher L means lighter colors (connective tissue); lower L means darker (parenchyma)
        L_values = centers[:, 0]
        sorted_idx = np.argsort(L_values)
        # Map cluster indices to semantic labels
        semantic_masks = {
            'parenchyma': np.zeros((H * W,), dtype=np.uint8),
            'connective': np.zeros((H * W,), dtype=np.uint8),
            'other': np.zeros((H * W,), dtype=np.uint8)
        }
        if self.n_clusters >= 3:
            parenchyma_cluster = sorted_idx[0]
            connective_cluster = sorted_idx[-1]
            other_clusters = sorted_idx[1:-1]
            semantic_masks['parenchyma'][labels == parenchyma_cluster] = 1
            semantic_masks['connective'][labels == connective_cluster] = 1
            for oc in other_clusters:
                semantic_masks['other'][labels == oc] = 1
        else:
            # Fallback: assign brightest cluster to connective, darkest to parenchyma
            parenchyma_cluster = sorted_idx[0]
            connective_cluster = sorted_idx[-1]
            semantic_masks['parenchyma'][labels == parenchyma_cluster] = 1
            semantic_masks['connective'][labels == connective_cluster] = 1
        # Reshape masks back to image shape
        for key in semantic_masks:
            semantic_masks[key] = semantic_masks[key].reshape(H, W)
        return semantic_masks
