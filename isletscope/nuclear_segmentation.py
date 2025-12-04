"""Enhanced nuclear segmentation for brightfield histology images.

This module provides specialized preprocessing and segmentation methods
optimized for nuclear detection in H&E-stained brightfield images. Unlike
the generic color deconvolution + InstanSeg approach, this module:

1. **Extracts the hematoxylin channel** using proper stain separation
   (not just normalization)
2. **Enhances nuclear contrast** via CLAHE and morphological operations
3. **Provides multiple segmentation backends** optimized for nuclei

Key improvements over the standard approach:
- True color deconvolution extracts nuclear-specific signal
- CLAHE adapts to local contrast variations
- Morphological enhancement boosts round nuclear structures
- Watershed with distance markers prevents over-segmentation
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Literal

import cv2
import numpy as np

try:
    from scipy import ndimage
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage.morphology import remove_small_objects, remove_small_holes
    from skimage.filters import threshold_otsu, threshold_local
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


# Standard H&E stain vectors (optical density space)
# These are well-established reference vectors for H&E stains
HE_STAIN_MATRIX = np.array([
    [0.644211, 0.716556, 0.266844],  # Hematoxylin
    [0.092789, 0.954111, 0.283111],  # Eosin
    [0.0, 0.0, 0.0]                  # Residual (computed via cross product)
]).T

# Alternative stain matrices for different preparations
STAIN_MATRICES = {
    'he_standard': np.array([
        [0.644211, 0.716556, 0.266844],  # Hematoxylin
        [0.092789, 0.954111, 0.283111],  # Eosin
    ]).T,
    'he_ruifrok': np.array([
        [0.650, 0.704, 0.286],  # Hematoxylin (Ruifrok & Johnston)
        [0.072, 0.990, 0.105],  # Eosin
    ]).T,
    'dab': np.array([
        [0.650, 0.704, 0.286],  # Hematoxylin
        [0.268, 0.570, 0.776],  # DAB
    ]).T,
}


class NuclearPreprocessor:
    """Preprocess brightfield H&E images for optimal nuclear segmentation.

    This class implements a multi-stage preprocessing pipeline specifically
    designed to enhance nuclear visibility before segmentation:

    1. **Hematoxylin extraction**: True color deconvolution to isolate nuclei
    2. **Contrast enhancement**: CLAHE to boost local contrast
    3. **Morphological enhancement**: Top-hat filtering for round structures
    4. **Noise reduction**: Bilateral filtering to smooth while preserving edges

    Example:
        >>> preprocessor = NuclearPreprocessor(stain_matrix='he_standard')
        >>> hematoxylin = preprocessor.extract_hematoxylin(image)
        >>> enhanced = preprocessor.enhance_nuclei(hematoxylin)
    """

    def __init__(
        self,
        stain_matrix: str = 'he_standard',
        clahe_clip_limit: float = 3.0,
        clahe_tile_size: int = 8,
        enhance_morphology: bool = True,
        morphology_kernel_size: int = 15,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75,
        bilateral_sigma_space: float = 75,
    ) -> None:
        """Initialize the nuclear preprocessor.

        Args:
            stain_matrix: Stain matrix to use for deconvolution.
                Options: 'he_standard', 'he_ruifrok', 'dab', or custom (3x2 array).
            clahe_clip_limit: CLAHE clip limit for contrast enhancement.
                Higher values = more contrast. Default: 3.0
            clahe_tile_size: CLAHE tile grid size. Default: 8
            enhance_morphology: Apply morphological enhancement (white top-hat).
            morphology_kernel_size: Kernel size for morphological operations.
            bilateral_d: Diameter for bilateral filter (noise reduction).
            bilateral_sigma_color: Color sigma for bilateral filter.
            bilateral_sigma_space: Space sigma for bilateral filter.
        """
        if isinstance(stain_matrix, str):
            if stain_matrix not in STAIN_MATRICES:
                raise ValueError(f"Unknown stain matrix: {stain_matrix}. "
                               f"Options: {list(STAIN_MATRICES.keys())}")
            self.stain_matrix = STAIN_MATRICES[stain_matrix]
        else:
            self.stain_matrix = np.asarray(stain_matrix)

        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.enhance_morphology = enhance_morphology
        self.morphology_kernel_size = morphology_kernel_size
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

        # Precompute deconvolution matrix
        self._setup_deconvolution_matrix()

    def _setup_deconvolution_matrix(self) -> None:
        """Set up the deconvolution matrix for stain separation."""
        # Normalize stain vectors
        stain_matrix = self.stain_matrix.copy()
        for i in range(stain_matrix.shape[1]):
            norm = np.linalg.norm(stain_matrix[:, i])
            if norm > 0:
                stain_matrix[:, i] /= norm

        # Add third vector via cross product if needed
        if stain_matrix.shape[1] == 2:
            v3 = np.cross(stain_matrix[:, 0], stain_matrix[:, 1])
            v3 /= np.linalg.norm(v3) + 1e-8
            stain_matrix = np.column_stack([stain_matrix, v3])

        # Compute pseudo-inverse for deconvolution
        self._deconv_matrix = np.linalg.pinv(stain_matrix)

    def rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to optical density (OD) space.

        Args:
            image: RGB image (H, W, 3), uint8 or float [0, 255].

        Returns:
            OD image (H, W, 3) in float32.
        """
        image = image.astype(np.float32)
        # Avoid log(0) by clamping minimum
        image = np.maximum(image, 1.0)
        # Convert to OD: -log(I/I0) where I0=255
        od = -np.log(image / 255.0)
        return od

    def od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert optical density back to RGB.

        Args:
            od: OD image (H, W, 3) in float32.

        Returns:
            RGB image (H, W, 3) in uint8.
        """
        rgb = np.exp(-od) * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def extract_hematoxylin(
        self,
        image: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Extract the hematoxylin (nuclear) channel via color deconvolution.

        This performs true stain separation, not just normalization. The
        hematoxylin channel specifically captures nuclear content.

        Args:
            image: RGB image (H, W, 3) in BGR or RGB format.
            normalize: Normalize output to [0, 255] range.

        Returns:
            Hematoxylin channel (H, W) as uint8 grayscale.
            Higher values = more hematoxylin = more nuclear content.
        """
        # Convert to OD space
        od = self.rgb_to_od(image)

        # Reshape for matrix multiplication: (H*W, 3)
        h, w = od.shape[:2]
        od_flat = od.reshape(-1, 3)

        # Deconvolve to get stain concentrations: (H*W, 3)
        # Result: [hematoxylin, eosin, residual] concentrations
        concentrations = od_flat @ self._deconv_matrix.T

        # Extract hematoxylin concentration (first channel)
        hematoxylin = concentrations[:, 0].reshape(h, w)

        # Clip negative values (can occur due to noise)
        hematoxylin = np.maximum(hematoxylin, 0)

        if normalize:
            # Normalize to [0, 255] for visualization and further processing
            hmax = np.percentile(hematoxylin, 99.5)
            if hmax > 0:
                hematoxylin = (hematoxylin / hmax) * 255.0
            hematoxylin = np.clip(hematoxylin, 0, 255).astype(np.uint8)

        return hematoxylin

    def extract_eosin(
        self,
        image: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Extract the eosin (cytoplasm/background) channel.

        Args:
            image: RGB image (H, W, 3).
            normalize: Normalize output to [0, 255] range.

        Returns:
            Eosin channel (H, W) as uint8 grayscale.
        """
        od = self.rgb_to_od(image)
        h, w = od.shape[:2]
        od_flat = od.reshape(-1, 3)
        concentrations = od_flat @ self._deconv_matrix.T

        eosin = concentrations[:, 1].reshape(h, w)
        eosin = np.maximum(eosin, 0)

        if normalize:
            emax = np.percentile(eosin, 99.5)
            if emax > 0:
                eosin = (eosin / emax) * 255.0
            eosin = np.clip(eosin, 0, 255).astype(np.uint8)

        return eosin

    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: Optional[float] = None,
        tile_size: Optional[int] = None,
    ) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        CLAHE enhances local contrast while limiting noise amplification,
        making it ideal for images with varying illumination.

        Args:
            image: Grayscale image (H, W) in uint8.
            clip_limit: Override default clip limit.
            tile_size: Override default tile grid size.

        Returns:
            Contrast-enhanced image (H, W) in uint8.
        """
        clip = clip_limit if clip_limit is not None else self.clahe_clip_limit
        tile = tile_size if tile_size is not None else self.clahe_tile_size

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return clahe.apply(image)

    def apply_morphological_enhancement(
        self,
        image: np.ndarray,
        kernel_size: Optional[int] = None,
    ) -> np.ndarray:
        """Apply morphological enhancement to boost round nuclear structures.

        Uses white top-hat transform to enhance bright structures (nuclei)
        that are smaller than the structuring element.

        Args:
            image: Grayscale image (H, W) in uint8.
            kernel_size: Size of structuring element (should be larger than
                typical nucleus diameter).

        Returns:
            Enhanced image (H, W) in uint8.
        """
        ksize = kernel_size if kernel_size is not None else self.morphology_kernel_size

        # Create circular structuring element
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # White top-hat: enhances bright peaks smaller than kernel
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

        # Combine with original (weighted sum)
        enhanced = cv2.addWeighted(image, 0.7, tophat, 0.3, 0)

        return enhanced

    def apply_bilateral_filter(
        self,
        image: np.ndarray,
        d: Optional[int] = None,
        sigma_color: Optional[float] = None,
        sigma_space: Optional[float] = None,
    ) -> np.ndarray:
        """Apply bilateral filter for edge-preserving noise reduction.

        Bilateral filtering smooths homogeneous regions while preserving
        edges (nuclear boundaries).

        Args:
            image: Grayscale image (H, W) in uint8.
            d: Diameter of pixel neighborhood.
            sigma_color: Filter sigma in the color space.
            sigma_space: Filter sigma in the coordinate space.

        Returns:
            Smoothed image (H, W) in uint8.
        """
        d = d if d is not None else self.bilateral_d
        sc = sigma_color if sigma_color is not None else self.bilateral_sigma_color
        ss = sigma_space if sigma_space is not None else self.bilateral_sigma_space

        return cv2.bilateralFilter(image, d, sc, ss)

    def enhance_nuclei(
        self,
        hematoxylin: np.ndarray,
        apply_clahe: bool = True,
        apply_morphology: bool = None,
        apply_bilateral: bool = True,
    ) -> np.ndarray:
        """Apply full enhancement pipeline to hematoxylin channel.

        Args:
            hematoxylin: Hematoxylin channel (H, W) from extract_hematoxylin().
            apply_clahe: Apply CLAHE contrast enhancement.
            apply_morphology: Apply morphological enhancement. If None, uses
                self.enhance_morphology.
            apply_bilateral: Apply bilateral filter for noise reduction.

        Returns:
            Enhanced grayscale image (H, W) in uint8.
        """
        enhanced = hematoxylin.copy()

        if apply_bilateral:
            enhanced = self.apply_bilateral_filter(enhanced)

        if apply_clahe:
            enhanced = self.apply_clahe(enhanced)

        use_morph = apply_morphology if apply_morphology is not None else self.enhance_morphology
        if use_morph:
            enhanced = self.apply_morphological_enhancement(enhanced)

        return enhanced

    def preprocess(
        self,
        image: np.ndarray,
        return_intermediate: bool = False,
    ) -> np.ndarray | Dict[str, np.ndarray]:
        """Full preprocessing pipeline for nuclear segmentation.

        Combines hematoxylin extraction + enhancement in one call.

        Args:
            image: RGB image (H, W, 3) in BGR format.
            return_intermediate: If True, return dict with all intermediate
                results for visualization.

        Returns:
            If return_intermediate=False: Enhanced grayscale image (H, W).
            If return_intermediate=True: Dict with 'hematoxylin', 'eosin',
                'enhanced', and 'rgb_nuclear' keys.
        """
        hematoxylin = self.extract_hematoxylin(image)
        enhanced = self.enhance_nuclei(hematoxylin)

        if not return_intermediate:
            return enhanced

        # Also extract eosin for reference
        eosin = self.extract_eosin(image)

        # Create pseudo-RGB for InstanSeg (expects 3 channels)
        rgb_nuclear = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return {
            'hematoxylin': hematoxylin,
            'eosin': eosin,
            'enhanced': enhanced,
            'rgb_nuclear': rgb_nuclear,
        }

    def to_pseudo_rgb(self, grayscale: np.ndarray) -> np.ndarray:
        """Convert grayscale to pseudo-RGB for models expecting 3 channels.

        Args:
            grayscale: Single-channel image (H, W) in uint8.

        Returns:
            Pseudo-RGB image (H, W, 3) in uint8.
        """
        return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)


class EnhancedNuclearSegmenter:
    """Enhanced nuclear segmentation for brightfield H&E images.

    This class combines the NuclearPreprocessor with multiple segmentation
    backends optimized for nuclear detection:

    1. **instanseg_hematoxylin**: Hematoxylin extraction → CLAHE → InstanSeg
    2. **watershed**: Distance transform + watershed (robust, no ML)
    3. **adaptive_threshold**: Adaptive thresholding + morphology

    Example:
        >>> segmenter = EnhancedNuclearSegmenter(backend='instanseg_hematoxylin')
        >>> result = segmenter.segment(image)
        >>> labels = result['labels']
    """

    def __init__(
        self,
        backend: Literal['instanseg_hematoxylin', 'watershed', 'adaptive_threshold'] = 'watershed',
        preprocessor: Optional[NuclearPreprocessor] = None,
        # Preprocessing parameters
        stain_matrix: str = 'he_standard',
        clahe_clip_limit: float = 3.0,
        # InstanSeg parameters
        instanseg_model_name: str = 'brightfield_nuclei',
        tile_size: int = 1024,
        tile_overlap: int = 64,
        # Watershed parameters
        min_distance: int = 10,
        threshold_method: str = 'otsu',
        # Common parameters
        min_size: int = 32,
        max_size: int = 5000,
        fill_holes: bool = True,
    ) -> None:
        """Initialize the enhanced nuclear segmenter.

        Args:
            backend: Segmentation method to use.
                - 'instanseg_hematoxylin': Preprocessed InstanSeg (best quality)
                - 'watershed': Distance transform watershed (robust, fast)
                - 'adaptive_threshold': Simple thresholding (fastest)
            preprocessor: Optional custom NuclearPreprocessor instance.
            stain_matrix: Stain matrix for hematoxylin extraction.
            clahe_clip_limit: CLAHE clip limit for contrast enhancement.
            instanseg_model_name: InstanSeg model for 'instanseg_hematoxylin' backend.
            tile_size: Tile size for InstanSeg processing.
            tile_overlap: Tile overlap for InstanSeg.
            min_distance: Minimum distance between nuclei for watershed.
            threshold_method: Thresholding method ('otsu' or 'adaptive').
            min_size: Minimum nucleus size in pixels.
            max_size: Maximum nucleus size in pixels.
            fill_holes: Fill holes in segmented nuclei.
        """
        self.backend = backend
        self.min_size = min_size
        self.max_size = max_size
        self.fill_holes = fill_holes
        self.min_distance = min_distance
        self.threshold_method = threshold_method

        # InstanSeg parameters
        self.instanseg_model_name = instanseg_model_name
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self._instanseg_model = None

        # Initialize preprocessor
        if preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            self.preprocessor = NuclearPreprocessor(
                stain_matrix=stain_matrix,
                clahe_clip_limit=clahe_clip_limit,
            )

    def segment(
        self,
        image: np.ndarray,
        tissue_mask: Optional[np.ndarray] = None,
        detect_tissue_first: bool = True,
        return_intermediate: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Segment nuclei in a brightfield H&E image.

        Args:
            image: RGB image (H, W, 3) in BGR format.
            tissue_mask: Optional pre-computed tissue mask.
            detect_tissue_first: Auto-detect tissue boundaries.
            return_intermediate: Include preprocessing results in output.

        Returns:
            Dict with 'mask', 'labels', and optionally preprocessing results.
        """
        # Detect tissue if requested
        if detect_tissue_first and tissue_mask is None:
            tissue_mask = self._detect_tissue(image)

        # Preprocess image
        prep_result = self.preprocessor.preprocess(image, return_intermediate=True)
        enhanced = prep_result['enhanced']

        # Select segmentation backend
        if self.backend == 'instanseg_hematoxylin':
            result = self._segment_instanseg(prep_result['rgb_nuclear'])
        elif self.backend == 'watershed':
            result = self._segment_watershed(enhanced)
        elif self.backend == 'adaptive_threshold':
            result = self._segment_adaptive(enhanced)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # Apply tissue mask
        if tissue_mask is not None:
            result = self._apply_tissue_mask(result, tissue_mask)

        # Post-process
        result = self._postprocess(result)

        # Include tissue mask and preprocessing results
        result['tissue_mask'] = tissue_mask
        if return_intermediate:
            result['preprocessing'] = prep_result

        return result

    def _detect_tissue(self, image: np.ndarray) -> np.ndarray:
        """Detect tissue regions vs. background/glass."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, tissue_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

        return (tissue_mask > 0).astype(np.uint8)

    def _segment_instanseg(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment using InstanSeg on preprocessed hematoxylin image."""
        try:
            from instanseg import InstanSeg
        except ImportError:
            print("InstanSeg not available, falling back to watershed")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self._segment_watershed(gray)

        # Lazy initialization
        if self._instanseg_model is None:
            self._instanseg_model = InstanSeg(
                self.instanseg_model_name,
                verbosity=0,
            )

        # Check if tiling needed
        h, w = image.shape[:2]
        if h > self.tile_size or w > self.tile_size:
            return self._segment_instanseg_tiled(image)

        # Single image inference
        if hasattr(self._instanseg_model, 'eval_small_image'):
            labeled, _ = self._instanseg_model.eval_small_image(image, None)
        else:
            labeled = self._instanseg_model.eval(image)
            if isinstance(labeled, tuple):
                labeled = labeled[0]

        labels = np.asarray(labeled)
        while labels.ndim > 2:
            labels = np.squeeze(labels, axis=0)

        labels = labels.astype(np.int32)
        mask = (labels > 0).astype(np.uint8)

        return {'mask': mask, 'labels': labels}

    def _segment_instanseg_tiled(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Tile-based InstanSeg for large images."""
        h, w = image.shape[:2]
        stride = self.tile_size - self.tile_overlap

        full_labels = np.zeros((h, w), dtype=np.int32)
        current_label = 1

        y_positions = list(range(0, h - self.tile_overlap, stride))
        x_positions = list(range(0, w - self.tile_overlap, stride))

        if y_positions[-1] + self.tile_size < h:
            y_positions.append(h - self.tile_size)
        if x_positions[-1] + self.tile_size < w:
            x_positions.append(w - self.tile_size)

        for y in y_positions:
            for x in x_positions:
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)

                tile = image[y:y_end, x:x_end]

                try:
                    if hasattr(self._instanseg_model, 'eval_small_image'):
                        labeled, _ = self._instanseg_model.eval_small_image(tile, None)
                    else:
                        labeled = self._instanseg_model.eval(tile)
                        if isinstance(labeled, tuple):
                            labeled = labeled[0]

                    tile_labels = np.asarray(labeled)
                    while tile_labels.ndim > 2:
                        tile_labels = np.squeeze(tile_labels, axis=0)

                    # Relabel and merge
                    for old_label in np.unique(tile_labels):
                        if old_label == 0:
                            continue
                        mask = tile_labels == old_label
                        full_labels[y:y_end, x:x_end][mask] = current_label
                        current_label += 1

                except Exception as e:
                    print(f"Warning: Tile at ({y}, {x}) failed: {e}")
                    continue

        # Re-label connected components
        mask = (full_labels > 0).astype(np.uint8)
        _, labels = cv2.connectedComponents(mask, connectivity=8)

        return {'mask': mask, 'labels': labels}

    def _segment_watershed(self, enhanced: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment using distance transform + watershed.

        This is a robust classical method that doesn't require ML models.
        """
        if not _SCIPY_AVAILABLE or not _SKIMAGE_AVAILABLE:
            # Fallback to simple connected components
            return self._segment_adaptive(enhanced)

        # Threshold to get binary mask
        if self.threshold_method == 'adaptive':
            # Adaptive thresholding for varying illumination
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 51, -5
            )
        else:
            # Otsu thresholding
            _, binary = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        # Clean up binary mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Distance transform
        dist = ndimage.distance_transform_edt(binary)

        # Find local maxima as markers
        # Normalize distance for consistent peak finding
        dist_norm = dist / (dist.max() + 1e-8)

        # Find peaks with minimum distance constraint
        local_max = peak_local_max(
            dist_norm,
            min_distance=self.min_distance,
            threshold_abs=0.1,
            labels=binary,
        )

        # Create markers
        markers = np.zeros_like(binary, dtype=np.int32)
        for i, (y, x) in enumerate(local_max, start=1):
            markers[y, x] = i

        # Dilate markers slightly for better watershed seeds
        markers = cv2.dilate(
            markers.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ).astype(np.int32)

        # Watershed
        labels = watershed(-dist, markers, mask=binary)

        mask = (labels > 0).astype(np.uint8)

        return {'mask': mask, 'labels': labels.astype(np.int32)}

    def _segment_adaptive(self, enhanced: np.ndarray) -> Dict[str, np.ndarray]:
        """Simple adaptive thresholding + connected components.

        Fastest method, good for well-stained samples.
        """
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 51, -5
        )

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Connected components
        _, labels = cv2.connectedComponents(binary, connectivity=8)
        mask = (labels > 0).astype(np.uint8)

        return {'mask': mask, 'labels': labels.astype(np.int32)}

    def _apply_tissue_mask(
        self,
        result: Dict[str, np.ndarray],
        tissue_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Apply tissue mask to remove background detections."""
        result['labels'] = result['labels'] * tissue_mask
        result['mask'] = result['mask'] * tissue_mask

        # Re-label to remove gaps
        _, result['labels'] = cv2.connectedComponents(
            result['mask'], connectivity=8
        )

        return result

    def _postprocess(self, result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Post-process segmentation results."""
        labels = result['labels']
        mask = result['mask']

        # Filter by size
        new_labels = np.zeros_like(labels)
        new_mask = np.zeros_like(mask)
        current_id = 1

        for label_id in np.unique(labels):
            if label_id == 0:
                continue

            component = labels == label_id
            size = component.sum()

            if size < self.min_size or size > self.max_size:
                continue

            # Fill holes if requested
            if self.fill_holes and _SKIMAGE_AVAILABLE:
                component = remove_small_holes(component, area_threshold=64)

            new_labels[component] = current_id
            new_mask[component] = 1
            current_id += 1

        result['labels'] = new_labels
        result['mask'] = new_mask

        return result


def segment_nuclei_brightfield(
    image: np.ndarray,
    backend: str = 'watershed',
    stain_matrix: str = 'he_standard',
    min_size: int = 32,
    detect_tissue: bool = True,
    return_preprocessing: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Convenience function for brightfield nuclear segmentation.

    Args:
        image: RGB image (H, W, 3) in BGR format.
        backend: 'instanseg_hematoxylin', 'watershed', or 'adaptive_threshold'.
        stain_matrix: Stain matrix for hematoxylin extraction.
        min_size: Minimum nucleus size in pixels.
        detect_tissue: Auto-detect tissue boundaries.
        return_preprocessing: Include preprocessing results.
        **kwargs: Additional arguments passed to EnhancedNuclearSegmenter.

    Returns:
        Dict with 'mask', 'labels', and optionally 'preprocessing'.

    Example:
        >>> result = segment_nuclei_brightfield(image, backend='watershed')
        >>> num_nuclei = result['labels'].max()
        >>> print(f"Detected {num_nuclei} nuclei")
    """
    segmenter = EnhancedNuclearSegmenter(
        backend=backend,
        stain_matrix=stain_matrix,
        min_size=min_size,
        **kwargs,
    )

    return segmenter.segment(
        image,
        detect_tissue_first=detect_tissue,
        return_intermediate=return_preprocessing,
    )
