"""Stain normalization utilities.

This module implements automated estimation and normalization of stain vectors
for histological images.  The primary class, :class:`StainNormalizer`,
provides methods to estimate stain matrices using the Macenko algorithm and
to normalize images to a reference stain matrix.  These functions work on
RGB images stored as NumPy arrays.

The implementation is adapted from common open‑source routines for color
deconvolution and is designed to run efficiently on CPU or GPU via
NumPy/CuPy.  No manual selection of regions is required.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Optional, Tuple


def _rgb_to_od(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image to optical density (OD) space.

    Optical density is computed as ``OD = -log((I + 1) / 255)``, where ``I``
    is the image intensity.  A small constant is added to avoid taking
    the logarithm of zero.

    Args:
        img: RGB image as a NumPy array of shape ``(H, W, 3)`` with
            values in ``[0, 255]``.

    Returns:
        Array of shape ``(H*W, 3)`` containing OD vectors.
    """
    # Flatten image and convert to floats in range [0,1]
    img = img.astype(np.float32) + 1.0  # avoid log(0)
    img_flat = img.reshape(-1, 3)
    od = -np.log(img_flat / 255.0)
    return od


def _od_to_rgb(od: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    """Convert optical density values back to RGB image.

    Args:
        od: Optical density values of shape ``(H*W, 3)``.
        shape: Original image shape ``(H, W, 3)``.

    Returns:
        Reconstructed RGB image of shape ``shape`` with values in ``[0,255]``.
    """
    img_flat = np.exp(-od) * 255.0
    img = img_flat.reshape(shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


class StainNormalizer:
    """Estimate and normalize stain vectors for histological images.

    The normalizer estimates a stain matrix using the Macenko algorithm and
    applies normalization by projecting the optical density (OD) values
    onto a canonical stain basis.  If a target matrix is provided, the
    image is normalized to that reference; otherwise the computed matrix
    is used as its own reference.
    """

    def __init__(self, alpha: float = 0.1, low_percentile: float = 1.0, high_percentile: float = 99.0) -> None:
        """Initialize the normalizer.

        Args:
            alpha: Parameter controlling the maximum angle for selecting stain
                vectors.  Lower values yield more aggressive removal of
                outliers.
            low_percentile: Percentile for the low bound when computing
                extreme projections.
            high_percentile: Percentile for the high bound when computing
                extreme projections.
        """
        self.alpha = alpha
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.stain_matrix_: Optional[np.ndarray] = None

    def estimate_stain_matrix(self, img: np.ndarray) -> np.ndarray:
        """Estimate the stain matrix using the Macenko algorithm.

        Args:
            img: RGB image as a NumPy array of shape ``(H, W, 3)``.

        Returns:
            A ``3x3`` matrix where each column corresponds to an estimated stain
            vector.  The first two columns span the stain space; the third
            column is the cross product to form a complete basis.
        """
        od = _rgb_to_od(img)
        # Remove pixels with very low optical density (background)
        mask = np.any(od > self.alpha, axis=1)
        od_filtered = od[mask]
        # Perform singular value decomposition on covariance matrix
        _, eigvecs = np.linalg.eigh(np.cov(od_filtered.T))
        # Sort eigenvectors by descending eigenvalues
        eigvecs = eigvecs[:, ::-1]
        # Project OD values onto the first two principal components
        projected = np.dot(od_filtered, eigvecs[:, :2])
        # Compute angles of projections
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        # Select extreme angles at given percentiles
        min_angle = np.percentile(angles, self.low_percentile)
        max_angle = np.percentile(angles, self.high_percentile)
        v1 = np.dot(eigvecs[:, :2], np.array([np.cos(min_angle), np.sin(min_angle)]))
        v2 = np.dot(eigvecs[:, :2], np.array([np.cos(max_angle), np.sin(max_angle)]))
        # Normalize stain vectors to unit length
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        # Force consistent ordering: put hematoxylin first (dominant blue)
        if v1[0] > v2[0]:
            stain_matrix = np.stack([v1, v2], axis=1)
        else:
            stain_matrix = np.stack([v2, v1], axis=1)
        # Third vector completes the orthonormal basis
        v3 = np.cross(stain_matrix[:, 0], stain_matrix[:, 1])
        stain_matrix = np.column_stack((stain_matrix, v3))
        self.stain_matrix_ = stain_matrix
        return stain_matrix

    def normalize(self, img: np.ndarray, target_matrix: Optional[np.ndarray] = None, ref_intensity: float = 240.0) -> np.ndarray:
        """Normalize the staining of an image to a reference stain matrix.

        Args:
            img: RGB image as a NumPy array of shape ``(H, W, 3)``.
            target_matrix: Optional ``3x3`` matrix representing the desired stain
                basis.  If ``None`` and the estimator has been fitted, the
                estimator's own stain matrix is used.
            ref_intensity: Reference intensity used when reconstructing the
                normalized image; values closer to 255 produce lighter
                normalization.

        Returns:
            Normalized RGB image as ``(H, W, 3)`` array.
        """
        if target_matrix is None:
            if self.stain_matrix_ is None:
                raise ValueError("Stain matrix not estimated; call estimate_stain_matrix first or provide target_matrix.")
            target_matrix = self.stain_matrix_
        # Convert image to OD space
        shape = img.shape
        od = _rgb_to_od(img)
        # Solve for concentration matrix C such that OD = W * C where W is stain matrix
        W = target_matrix[:, :2]
        # Use pseudoinverse for least squares solution
        C, _, _, _ = np.linalg.lstsq(W, od.T, rcond=None)
        # Reconstruct OD values using canonical stain vectors with fixed intensity
        maxC = np.percentile(C, 99, axis=1)
        C = C / maxC[:, np.newaxis] * (ref_intensity / 255.0)
        od_norm = np.dot(target_matrix[:, :2], C)
        # Convert back to RGB
        img_norm = _od_to_rgb(od_norm.T, shape)
        return img_norm
