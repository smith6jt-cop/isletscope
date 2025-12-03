"""Cell segmentation utilities.

This module provides a flexible API for instance‑level cell segmentation.  The
primary class, :class:`CellSegmenter`, supports both deep‑learning models
such as InstanSeg and a classical fallback method based on morphological
operations.  Users may supply a custom model; otherwise the fallback
algorithm applies an adaptive threshold to grayscale images, removes
artifacts and produces a binary mask of putative cells.

The segmentation output is a dictionary containing:

* ``mask`` – a binary mask of the same height and width as the input image
  where ``1`` indicates cell pixels.
* ``labels`` – a labeled array where each connected component is assigned a
  unique integer identifier (optional).
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Any, Dict, Optional

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class CellSegmenter:
    """Segment cells in brightfield or fluorescent images.

    When instantiated with a deep‑learning model, the segmenter will delegate
    segmentation to the model.  Otherwise, it falls back to a simple
    threshold‑based method that works reasonably well for nuclei stained
    images where stained nuclei appear darker than background.
    """

    def __init__(self, model: Optional[Any] = None, device: Optional[str] = None) -> None:
        """Initialize the segmenter.

        Args:
            model: Optional deep‑learning model with a callable interface
                ``model.predict(image: np.ndarray) -> np.ndarray``.  The
                returned array should contain a binary or multi‑class mask.
            device: Optional device string for PyTorch models (e.g. ``'cuda'``).
        """
        self.model = model
        self.device = device

        if model is not None and _TORCH_AVAILABLE and hasattr(model, 'to'):
            # Move model to requested device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = device
            model.to(device)

    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment cells in an image.

        Args:
            image: Input image as a NumPy array of shape ``(H, W, 3)`` or
                ``(H, W)``.  Color images will be converted to grayscale.

        Returns:
            A dictionary with keys ``'mask'`` and ``'labels'``.  The ``mask``
            is a binary array of shape ``(H, W)`` where ``1`` denotes cell
            pixels.  The ``labels`` array assigns a unique integer label to
            each connected cell region (or ``None`` if labeling is skipped).
        """
        if self.model is not None:
            return self._segment_with_model(image)
        else:
            return self._segment_classical(image)

    def _segment_with_model(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform segmentation using a deep‑learning model.

        Args:
            image: Input image as a NumPy array.

        Returns:
            A dictionary containing ``mask`` and ``labels`` arrays.  The
            ``mask`` is computed by thresholding the model output; the
            ``labels`` array is optional.
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model‑based segmentation but is not installed.")
        # Preprocess image for model
        img = image.astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = np.transpose(img, (2, 0, 1))  # channels first
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        if self.device:
            img_tensor = img_tensor.to(self.device)
        # Forward pass
        with torch.no_grad():
            pred = self.model(img_tensor)
        # Assume output is sigmoid probability map or multi‑channel mask
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred_np = pred.squeeze().cpu().numpy()
        # For multi‑channel output take argmax; for single‑channel apply threshold
        if pred_np.ndim == 3:
            labels = np.argmax(pred_np, axis=0).astype(np.int32)
            mask = (labels > 0).astype(np.uint8)
        else:
            mask = (pred_np > 0.5).astype(np.uint8)
            # Label connected components
            num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        return {'mask': mask, 'labels': labels}

    def _segment_classical(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform segmentation using classical image processing.

        The algorithm converts the image to grayscale, applies a Gaussian
        blur, uses Otsu's threshold to obtain a binary mask and refines the
        mask using morphological opening and closing.  Finally, connected
        components are labeled.

        Args:
            image: Input image.

        Returns:
            Dictionary with binary ``mask`` and labeled ``labels`` arrays.
        """
        # Convert to grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        # Apply Gaussian blur to smooth noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # In brightfield with hematoxylin, nuclei are darker than background
        # Invert mask so that cell nuclei are 1
        mask = (thresh == 0).astype(np.uint8)
        # Morphological opening to remove small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        # Closing to fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Label connected components
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        return {'mask': mask, 'labels': labels}
