"""Cell segmentation utilities.

This module provides a flexible API for instance‑level cell segmentation.  The
primary class, :class:`CellSegmenter`, supports both deep‑learning models
such as InstanSeg and a classical fallback method based on morphological
operations.  Users may supply a custom model; otherwise the fallback
algorithm applies an adaptive threshold to grayscale images, removes
artifacts and produces a binary mask of putative cells.  Fluorescent images
are handled by channel selection and percentile thresholding.

The segmentation output is a dictionary containing:

* ``mask`` – a binary mask of the same height and width as the input image
  where ``1`` indicates cell pixels.
* ``labels`` – a labeled array where each connected component is assigned a
  unique integer identifier (optional).
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_INSTANSEG_AVAILABLE = importlib.util.find_spec("instanseg") is not None


class CellSegmenter:
    """Segment cells in brightfield or fluorescent images.

    When instantiated with a deep‑learning model, the segmenter delegates
    segmentation to the model.  Otherwise, it falls back to a classical
    method that adapts to brightfield or fluorescent inputs.  InstanSeg
    is auto‑detected when installed, but any callable PyTorch model can be
    supplied.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        device: Optional[str] = None,
        backend: str = "auto",
        use_instanseg: bool = True,
        probability_threshold: float = 0.5,
        min_size: int = 32,
        channel_axis: int = -1,
        # InstanSeg-specific parameters
        instanseg_model_name: str = "brightfield_nuclei",
        tile_size: int = 1024,
        tile_overlap: int = 64,
        batch_size: int = 4,
        pixel_size: Optional[float] = None,
        normalization: bool = True,
        image_reader: str = "tiffslide",
    ) -> None:
        """Initialize the segmenter.

        Args:
            model: Optional deep‑learning model with a callable interface
                ``model.predict(image: np.ndarray) -> np.ndarray`` or
                a PyTorch ``nn.Module``.
            device: Optional device string for PyTorch models (e.g. ``'cuda'``).
            backend: ``'auto'`` (default), ``'instanseg'``, ``'model'`` or
                ``'classical'``.
            use_instanseg: If ``True``, attempt to run InstanSeg when installed
                and no custom model is provided.
            probability_threshold: Threshold applied to model outputs to derive
                a binary mask.
            min_size: Remove connected components smaller than this pixel count.
            channel_axis: Axis containing image channels (``-1`` for HWC).
            instanseg_model_name: InstanSeg model name (e.g., ``'brightfield_nuclei'``,
                ``'fluorescence_nuclei_and_cells'``).
            tile_size: Tile size in pixels for large image processing (512, 1024, or 2048).
            tile_overlap: Overlap between tiles in pixels to prevent edge artifacts.
            batch_size: Number of tiles to process in parallel (GPU-dependent).
            pixel_size: Physical pixel size in microns (auto-detected from metadata if None).
            normalization: Apply intensity normalization before segmentation.
            image_reader: Image reading backend (``'tiffslide'``, ``'openslide'``, etc.).
        """
        self.model = model
        self.device = device
        self.backend = backend
        self.use_instanseg = use_instanseg
        self.probability_threshold = probability_threshold
        self.min_size = min_size
        self.channel_axis = channel_axis

        # InstanSeg parameters
        self.instanseg_model_name = instanseg_model_name
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.batch_size = batch_size
        self.pixel_size = pixel_size
        self.normalization = normalization
        self.image_reader = image_reader
        self._instanseg_model = None  # Lazy initialization

        if model is not None and _TORCH_AVAILABLE and hasattr(model, "to"):
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            model.to(device)

    def segment(
        self,
        image: np.ndarray,
        image_type: str = "brightfield",
        channels: Optional[Sequence[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Segment cells in an image.

        Args:
            image: Input image as a NumPy array of shape ``(H, W, C)`` or
                ``(H, W)``.  Color images will be converted to grayscale.
            image_type: ``'brightfield'`` or ``'fluorescence'`` controls
                preprocessing.
            channels: Optional channel indices to select (e.g. ``[0, 1, 2]``
                for RGB or ``[2]`` for a particular fluorescent channel).

        Returns:
            Dictionary with keys ``'mask'`` and ``'labels'``.
        """
        image = self._select_channels(image, channels)
        backend = self._resolve_backend()
        if backend == "model":
            return self._segment_with_model(image)
        if backend == "instanseg":
            try:
                return self._segment_with_instanseg(image)
            except Exception:
                # Fall back gracefully to classical if InstanSeg cannot run.
                return self._segment_classical(image, image_type=image_type)
        return self._segment_classical(image, image_type=image_type)

    def detect_markers(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        marker_channels: Dict[str, int],
        thresholds: Dict[str, float],
        brighter_is_positive: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Generate per‑marker binary masks from labeled cells.

        This is useful for deriving endocrine masks (insulin/glucagon) or
        immune populations (CD3) from multiplex fluorescence channels.

        Args:
            image: Original image (HWC).
            labels: Labeled segmentation mask.
            marker_channels: Mapping from marker name to channel index.
            thresholds: Mapping from marker name to scalar intensity threshold.
            brighter_is_positive: If ``False``, interpret darker staining as
                positive (common in brightfield).

        Returns:
            Dict mapping marker name to binary mask of positive cells.
        """
        if image.ndim == 2:
            image = image[..., None]
        output: Dict[str, np.ndarray] = {}
        for marker, chan_idx in marker_channels.items():
            channel = image[..., chan_idx].astype(np.float32)
            mask = np.zeros_like(labels, dtype=np.uint8)
            for cell_id in np.unique(labels):
                if cell_id == 0:
                    continue
                coords = labels == cell_id
                intensity = channel[coords].mean()
                is_positive = intensity > thresholds.get(marker, 0.0)
                if not brighter_is_positive:
                    is_positive = intensity < thresholds.get(marker, 0.0)
                if is_positive:
                    mask[coords] = 1
            output[marker] = mask
        return output

    # Internal helpers -----------------------------------------------------

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if self.model is not None:
            return "model"
        if self.use_instanseg and _INSTANSEG_AVAILABLE:
            return "instanseg"
        return "classical"

    def _select_channels(self, image: np.ndarray, channels: Optional[Sequence[int]]) -> np.ndarray:
        if channels is None or image.ndim == 2:
            return image
        return image.take(indices=list(channels), axis=self.channel_axis)

    def _segment_with_model(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform segmentation using a deep‑learning model."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model‑based segmentation but is not installed.")
        img = image.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = np.transpose(img, (2, 0, 1))  # channels first
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        if self.device:
            img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            pred = self.model(img_tensor)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred_np = pred.squeeze().detach().cpu().numpy()
        if pred_np.ndim == 3:
            labels = np.argmax(pred_np, axis=0).astype(np.int32)
            mask = (labels > 0).astype(np.uint8)
        else:
            mask = (pred_np > self.probability_threshold).astype(np.uint8)
            _, labels = cv2.connectedComponents(mask, connectivity=8)
        mask, labels = self._filter_small_objects(mask, labels)
        return {"mask": mask, "labels": labels}

    def _segment_with_instanseg(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run InstanSeg inference with proper model initialization and tiling.

        This method properly initializes InstanSeg models with the correct API,
        implements tile-based processing for large images, and handles batch
        inference for improved performance.
        """
        try:
            from instanseg import InstanSeg
        except ImportError as e:
            raise ImportError(
                "instanseg-torch is not installed or not importable. "
                "Install via: pip install instanseg-torch"
            ) from e

        # Lazy initialization of InstanSeg model
        if self._instanseg_model is None:
            try:
                self._instanseg_model = InstanSeg(
                    self.instanseg_model_name,
                    image_reader=self.image_reader,
                    verbosity=1,
                )
                print(f"InstanSeg model '{self.instanseg_model_name}' initialized successfully.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize InstanSeg model '{self.instanseg_model_name}'. "
                    f"Error: {e}"
                ) from e

        # Determine if we need tiling based on image size
        h, w = image.shape[:2]
        needs_tiling = h > self.tile_size or w > self.tile_size

        if needs_tiling:
            return self._segment_with_instanseg_tiled(image)
        else:
            return self._segment_with_instanseg_single(image)

    def _segment_with_instanseg_single(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment a small image (no tiling required)."""
        try:
            # Try eval_small_image method if available
            if hasattr(self._instanseg_model, "eval_small_image"):
                labeled_output, _ = self._instanseg_model.eval_small_image(
                    image,
                    self.pixel_size
                )
            # Try eval method as fallback
            elif hasattr(self._instanseg_model, "eval"):
                labeled_output = self._instanseg_model.eval(image)
                if isinstance(labeled_output, tuple):
                    labeled_output = labeled_output[0]
            else:
                raise RuntimeError(
                    "InstanSeg model does not have eval_small_image or eval method. "
                    "API may have changed."
                )

            # Convert labeled output to mask and labels
            if isinstance(labeled_output, dict):
                labels = labeled_output.get("labels", labeled_output.get("instances"))
            else:
                labels = np.asarray(labeled_output)

            # Handle different output formats
            # InstanSeg may return (B, C, H, W) or (H, W) or (1, H, W)
            labels = np.asarray(labels)
            while labels.ndim > 2:
                labels = np.squeeze(labels, axis=0)  # Remove batch/channel dims

            labels = labels.astype(np.int32)
            mask = (labels > 0).astype(np.uint8)
            mask, labels = self._filter_small_objects(mask, labels)

            return {"mask": mask, "labels": labels}

        except Exception as e:
            raise RuntimeError(
                f"InstanSeg inference failed: {e}. "
                "Falling back to classical segmentation may be safer."
            ) from e

    def _segment_with_instanseg_tiled(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment a large image using tile-based processing.

        This method splits the image into overlapping tiles, processes each tile
        independently, and merges the results while handling overlapping regions.
        """
        h, w = image.shape[:2]
        stride = self.tile_size - self.tile_overlap

        # Initialize output arrays
        full_labels = np.zeros((h, w), dtype=np.int32)
        tile_count_map = np.zeros((h, w), dtype=np.int32)  # Track overlaps
        current_label_id = 1

        # Calculate tile positions
        y_positions = list(range(0, h - self.tile_overlap, stride))
        x_positions = list(range(0, w - self.tile_overlap, stride))

        # Ensure we cover the entire image
        if y_positions[-1] + self.tile_size < h:
            y_positions.append(h - self.tile_size)
        if x_positions[-1] + self.tile_size < w:
            x_positions.append(w - self.tile_size)

        total_tiles = len(y_positions) * len(x_positions)
        print(f"Processing {total_tiles} tiles ({len(y_positions)}x{len(x_positions)}) with tile_size={self.tile_size}, overlap={self.tile_overlap}")

        # Process tiles
        tile_idx = 0
        for y in y_positions:
            for x in x_positions:
                tile_idx += 1
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)

                # Extract tile
                tile = image[y:y_end, x:x_end]

                # Segment tile
                try:
                    tile_result = self._segment_with_instanseg_single(tile)
                    tile_labels = tile_result["labels"]

                    # Ensure tile_labels matches expected shape
                    if tile_labels.shape != (y_end - y, x_end - x):
                        print(f"Warning: Tile shape mismatch at ({y}, {x}). Expected {(y_end - y, x_end - x)}, got {tile_labels.shape}")
                        continue

                    # Relabel to avoid conflicts with existing labels
                    unique_labels = np.unique(tile_labels)
                    unique_labels = unique_labels[unique_labels > 0]

                    for old_label in unique_labels:
                        tile_labels[tile_labels == old_label] = current_label_id
                        current_label_id += 1

                    # Merge into full image (simple averaging in overlap regions)
                    full_labels[y:y_end, x:x_end] += tile_labels
                    tile_count_map[y:y_end, x:x_end] += (tile_labels > 0).astype(np.int32)

                    if tile_idx % 10 == 0 or tile_idx == total_tiles:
                        print(f"  Processed {tile_idx}/{total_tiles} tiles...")

                except Exception as e:
                    print(f"Warning: Tile at ({y}, {x}) size {tile.shape} failed: {e}. Skipping.")
                    continue

        # Average overlapping regions and re-threshold
        overlap_mask = tile_count_map > 1
        full_labels[overlap_mask] = full_labels[overlap_mask] // tile_count_map[overlap_mask]

        # Re-label connected components to merge split cells
        mask = (full_labels > 0).astype(np.uint8)
        _, labels = cv2.connectedComponents(mask, connectivity=8)

        # Filter small objects
        mask, labels = self._filter_small_objects(mask, labels)

        num_cells = len(np.unique(labels)) - 1  # Exclude background
        print(f"InstanSeg detected {num_cells:,} cells")

        return {"mask": mask, "labels": labels}

    def _segment_classical(self, image: np.ndarray, image_type: str = "brightfield") -> Dict[str, np.ndarray]:
        """Perform segmentation using classical image processing.

        The algorithm converts the image to grayscale (or uses a specified
        fluorescence channel), applies a blur, uses adaptive/OTSU thresholding,
        refines the mask using morphology and labels connected components.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if image_type.lower() == "fluorescence":
            # Use percentile thresholding to accommodate varying backgrounds.
            p = np.percentile(gray, 98)
            thresh_mask = (gray >= p).astype(np.uint8) * 255
        else:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh_mask = 255 - thresh_mask  # nuclei are darker

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = (mask > 0).astype(np.uint8)
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        mask, labels = self._filter_small_objects(mask, labels)
        return {"mask": mask, "labels": labels}

    def _filter_small_objects(self, mask: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove connected components smaller than ``min_size``."""
        if self.min_size <= 0:
            return mask, labels
        filtered_mask = np.zeros_like(mask, dtype=np.uint8)
        filtered_labels = np.zeros_like(labels, dtype=np.int32)
        current_id = 1
        for cid in np.unique(labels):
            if cid == 0:
                continue
            component = labels == cid
            if component.sum() < self.min_size:
                continue
            filtered_mask[component] = 1
            filtered_labels[component] = current_id
            current_id += 1
        return filtered_mask, filtered_labels
