"""Stain normalization utilities.

This module implements automated estimation and normalization of stain vectors
for histological images.  The primary class, :class:`StainNormalizer`,
provides methods to estimate stain matrices using the Macenko algorithm,
optionally perform a Vahadane‑style non‑negative matrix factorization, and to
normalize brightfield or fluorescent images without manual region selection.
GPU acceleration via CuPy is used when available and requested.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore

    _CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    _CUPY_AVAILABLE = False


def _get_array_module(use_gpu: bool):
    """Return NumPy or CuPy depending on availability and flag."""
    if use_gpu and _CUPY_AVAILABLE:
        return cp
    return np


def _rgb_to_od_backend(img: np.ndarray, xp) -> any:
    """Convert an RGB image to optical density (OD) space using the provided array module."""
    arr = xp.asarray(img, dtype=xp.float32) + 1.0  # avoid log(0)
    arr_flat = arr.reshape(-1, arr.shape[-1])
    od = -xp.log(arr_flat / 255.0)
    return od


def _od_to_rgb_backend(od, shape: Tuple[int, int, int], xp) -> np.ndarray:
    """Convert optical density values back to an RGB image using the provided array module."""
    img_flat = xp.exp(-od) * 255.0
    img = img_flat.reshape(shape)
    img = xp.clip(img, 0, 255)
    img_np = np.asarray(img.get() if xp is cp else img, dtype=np.uint8)
    return img_np


class StainNormalizer:
    """Estimate and normalize stain vectors for histological images.

    The normalizer estimates a stain matrix (Macenko by default) and applies
    normalization by projecting optical density (OD) values onto a canonical
    stain basis.  For fluorescent images, normalization falls back to robust
    per‑channel percentile scaling.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        low_percentile: float = 1.0,
        high_percentile: float = 99.0,
        use_gpu: bool = False,
        method: str = "macenko",
        fluorescence_percentiles: Tuple[float, float] = (1.0, 99.0),
        nmf_max_iter: int = 75,
    ) -> None:
        """Initialize the normalizer.

        Args:
            alpha: Parameter controlling the maximum angle for selecting stain
                vectors.  Lower values yield more aggressive removal of
                outliers.
            low_percentile: Percentile for the low bound when computing extreme
                projections (Macenko).
            high_percentile: Percentile for the high bound when computing
                extreme projections (Macenko).
            use_gpu: If ``True`` and CuPy is installed, run computations on GPU.
            method: ``"macenko"`` (default) or ``"vahadane"``.  Vahadane uses a
                lightweight NMF solver implemented here to avoid extra
                dependencies.
            fluorescence_percentiles: Lower/upper percentiles for scaling
                fluorescent channels.
            nmf_max_iter: Maximum iterations for the internal NMF solver when
                ``method="vahadane"``.
        """
        self.alpha = alpha
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.use_gpu = use_gpu
        self.method = method
        self.fluorescence_percentiles = fluorescence_percentiles
        self.nmf_max_iter = nmf_max_iter
        self.stain_matrix_: Optional[np.ndarray] = None

    def estimate_stain_matrix(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Estimate the stain matrix using the requested method.

        For brightfield images the Macenko method is used by default.  The
        optional Vahadane branch performs a simple NMF to derive stain vectors.
        Fluorescent images do not require a stain matrix; in that case the
        method returns ``None`` and per‑channel percentile scaling is used.
        """
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError("Expected an RGB image with 3 channels for stain estimation.")

        if self.method.lower() == "vahadane":
            stain_matrix = self._estimate_vahadane(img)
        else:
            stain_matrix = self._estimate_macenko(img)
        self.stain_matrix_ = stain_matrix
        return stain_matrix

    def _estimate_macenko(self, img: np.ndarray) -> np.ndarray:
        """Macenko stain estimation with optional GPU acceleration."""
        xp = _get_array_module(self.use_gpu)
        od = _rgb_to_od_backend(img, xp)
        mask = xp.any(od > self.alpha, axis=1)
        od_filtered = od[mask]
        if od_filtered.shape[0] < 10:
            raise ValueError("Not enough foreground pixels to estimate stain vectors.")
        _, eigvecs = xp.linalg.eigh(xp.cov(od_filtered.T))
        eigvecs = eigvecs[:, ::-1]
        projected = od_filtered @ eigvecs[:, :2]
        angles = xp.arctan2(projected[:, 1], projected[:, 0])
        min_angle = xp.percentile(angles, self.low_percentile)
        max_angle = xp.percentile(angles, self.high_percentile)
        v1 = eigvecs[:, :2] @ xp.array([xp.cos(min_angle), xp.sin(min_angle)])
        v2 = eigvecs[:, :2] @ xp.array([xp.cos(max_angle), xp.sin(max_angle)])
        v1 = v1 / (xp.linalg.norm(v1) + 1e-8)
        v2 = v2 / (xp.linalg.norm(v2) + 1e-8)
        if v1[0] > v2[0]:
            stain_matrix = xp.stack([v1, v2], axis=1)
        else:
            stain_matrix = xp.stack([v2, v1], axis=1)
        v3 = xp.cross(stain_matrix[:, 0], stain_matrix[:, 1])
        stain_matrix = xp.column_stack((stain_matrix, v3))
        return np.asarray(stain_matrix.get() if xp is cp else stain_matrix)

    def _estimate_vahadane(self, img: np.ndarray) -> np.ndarray:
        """Lightweight Vahadane‑style NMF to estimate stain matrix without extra deps."""
        xp = _get_array_module(self.use_gpu)
        od = _rgb_to_od_backend(img, xp)
        od = od[xp.any(od > self.alpha, axis=1)]
        if od.shape[0] < 10:
            raise ValueError("Not enough foreground pixels to estimate stain vectors.")
        rng = xp.random.default_rng(seed=0)
        W = xp.abs(rng.standard_normal((3, 2)))
        H = xp.abs(rng.standard_normal((2, od.shape[0])))
        beta = 1e-6
        for _ in range(self.nmf_max_iter):
            numerator_H = W.T @ od
            denominator_H = (W.T @ W @ H) + beta
            H *= numerator_H / denominator_H
            numerator_W = od @ H.T
            denominator_W = (W @ (H @ H.T)) + beta
            W *= numerator_W / denominator_W
            W = xp.maximum(W, beta)
        W = W / (xp.linalg.norm(W, axis=0, keepdims=True) + 1e-8)
        v3 = xp.cross(W[:, 0], W[:, 1])
        stain_matrix = xp.column_stack((W, v3))
        return np.asarray(stain_matrix.get() if xp is cp else stain_matrix)

    def normalize(
        self,
        img: np.ndarray,
        target_matrix: Optional[np.ndarray] = None,
        ref_intensity: float = 240.0,
        image_type: str = "brightfield",
    ) -> np.ndarray:
        """Normalize an image to a reference stain matrix or via percentile scaling.

        Args:
            img: Input image, brightfield RGB or multi‑channel fluorescence.
            target_matrix: Optional ``3x3`` matrix representing the desired stain
                basis.  If ``None`` and the estimator has been fitted, the
                estimator's own stain matrix is used.
            ref_intensity: Reference intensity used when reconstructing the
                normalized image; values closer to 255 produce lighter
                normalization.
            image_type: ``'brightfield'`` (default) uses stain deconvolution;
                ``'fluorescence'`` applies robust per‑channel percentile scaling.
        """
        if image_type.lower() == "fluorescence":
            return self._normalize_fluorescence(img)

        if target_matrix is None:
            if self.stain_matrix_ is None:
                raise ValueError("Stain matrix not estimated; call estimate_stain_matrix first or provide target_matrix.")
            target_matrix = self.stain_matrix_
        xp = _get_array_module(self.use_gpu)
        shape = img.shape
        od = _rgb_to_od_backend(img, xp)
        W = xp.asarray(target_matrix)[:, :2]
        C, _, _, _ = xp.linalg.lstsq(W, od.T, rcond=None)
        maxC = xp.percentile(C, 99, axis=1)
        maxC = xp.where(maxC == 0, 1.0, maxC)
        C = C / maxC[:, xp.newaxis] * (ref_intensity / 255.0)
        od_norm = W @ C
        img_norm = _od_to_rgb_backend(od_norm.T, shape, xp)
        return img_norm

    def _normalize_fluorescence(self, img: np.ndarray) -> np.ndarray:
        """Robust percentile scaling for multiplex fluorescent images."""
        if img.ndim == 2:
            img = img[..., None]
        img = img.astype(np.float32)
        lower, upper = self.fluorescence_percentiles
        scaled_channels = []
        for c in range(img.shape[2]):
            chan = img[..., c]
            lo = np.percentile(chan, lower)
            hi = np.percentile(chan, upper)
            hi = hi if hi > lo else lo + 1e-3
            norm = (chan - lo) / (hi - lo)
            norm = np.clip(norm, 0.0, 1.0)
            scaled_channels.append((norm * 255.0).astype(np.uint8))
        return np.stack(scaled_channels, axis=-1)
