"""3D anatomical inference from 2D pancreas sections.

This module estimates how a 2D section maps into the pancreas head–body–tail
axis and depth using lightweight heuristics derived from vessel density,
islet density and tissue composition.  The output is a coarse coordinate in
``[0, 1]^3`` suitable for downstream visualization; more advanced atlas
registration can be layered on top later.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


class SpatialInferer:
    """Infer approximate 3D coordinates of a 2D pancreas section."""

    def __init__(self) -> None:
        pass

    def infer(
        self,
        image: np.ndarray,
        vessel_mask: Optional[np.ndarray] = None,
        islet_mask: Optional[np.ndarray] = None,
        tissue_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Infer normalized 3D coordinates for a section.

        Args:
            image: RGB image of shape ``(H, W, 3)``.
            vessel_mask: Optional binary mask of vessels/ducts; if ``None`` a
                brightness threshold is used as an estimator.
            islet_mask: Optional binary mask of islets to estimate endocrine
                density.
            tissue_masks: Optional dictionary from tissue classification to
                refine depth (e.g. ``parenchyma`` vs ``connective``).

        Returns:
            Dictionary with keys ``'head_body_tail'``, ``'depth'``, ``'lateral'``
            and ``'vessel_fraction'``.
        """
        H, W, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Vessel/duct estimation
        if vessel_mask is None:
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            vessel_mask = thresh.astype(np.uint8)
        vessel_fraction = float(np.count_nonzero(vessel_mask)) / float(H * W)
        vessel_fraction = min(1.0, vessel_fraction)

        # Islet density estimation
        islet_fraction = 0.0
        if islet_mask is not None:
            islet_fraction = float(np.count_nonzero(islet_mask)) / float(H * W)

        # Tissue composition to refine depth
        parenchyma_fraction = connective_fraction = background_fraction = 0.0
        if tissue_masks:
            parenchyma_fraction = float(np.count_nonzero(tissue_masks.get("parenchyma", 0))) / float(H * W)
            connective_fraction = float(np.count_nonzero(tissue_masks.get("connective", 0))) / float(H * W)
            background_fraction = float(np.count_nonzero(tissue_masks.get("background", 0))) / float(H * W)
        else:
            background_fraction = float(np.count_nonzero(gray > 250)) / float(H * W)

        # Heuristic coordinates
        head_score = min(1.0, vessel_fraction * 3.0 + connective_fraction * 0.5)
        tail_score = min(1.0, islet_fraction * 3.0 + parenchyma_fraction * 0.5)
        denom = head_score + tail_score + 1e-6
        head_body_tail = tail_score / denom  # 0=head, 1=tail

        lateral = min(1.0, max(0.0, (W - H) / max(W, H) * 0.5 + 0.5))
        depth = min(1.0, max(0.0, 1.0 - (background_fraction * 2.5 + connective_fraction)))

        return {
            "head_body_tail": float(head_body_tail),
            "depth": float(depth),
            "lateral": float(lateral),
            "vessel_fraction": float(vessel_fraction),
            "islet_fraction": float(islet_fraction),
        }
