Tutorial
========

This tutorial provides a high‑level overview of the IsletScope workflow.  For
practical examples see the Jupyter notebooks in the repository under
``notebooks/``.  The workflow is divided into six stages and supports both
brightfield and multiplex fluorescence images.

Stain normalization
-------------------

Images acquired on different scanners or at different times often exhibit
variation in staining intensity and hue.  The :class:`isletscope.stain.StainNormalizer`
implements automated Macenko or Vahadane estimation for brightfield along with
GPU acceleration (via CuPy when available).  Fluorescent images use robust
per‑channel percentile scaling instead of stain deconvolution.

.. code-block:: python

   from isletscope.stain import StainNormalizer
   import cv2

   img = cv2.imread('sample.tif')
   normalizer = StainNormalizer(method="macenko", use_gpu=True)
   normalizer.estimate_stain_matrix(img)
   img_norm = normalizer.normalize(img)

For multiplex fluorescence:

.. code-block:: python

   normalizer = StainNormalizer()
   img_norm = normalizer.normalize(fluorescent_img, image_type="fluorescence")

Environment tip: use ``environment.yml`` (GPU) or ``environment.cpu.yml`` (CPU) with ``mamba env create -f ...`` before running the notebooks.


Cell segmentation
-----------------

Segmenting individual cells is the foundation for downstream analysis.  Use
:class:`isletscope.segmentation.CellSegmenter` to perform segmentation via
a deep‑learning model (e.g., InstanSeg) or a classical thresholding fallback:

.. code-block:: python

   from isletscope.segmentation import CellSegmenter
   segmenter = CellSegmenter(backend="auto", use_instanseg=True)
   result = segmenter.segment(img_norm, image_type="brightfield")
   mask = result['mask']
   labels = result['labels']

Marker detection for multiplex images is available via ``detect_markers``:

.. code-block:: python

   markers = segmenter.detect_markers(img_norm, labels,
                                      marker_channels={'insulin': 0, 'glucagon': 1, 'CD3': 2},
                                      thresholds={'insulin': 80, 'glucagon': 80, 'CD3': 40},
                                      brighter_is_positive=True)


Islet detection
---------------

Islet detection classifies endocrine clusters into islets or single cells.
Instantiate :class:`isletscope.islet_detection.IsletDetector` and call
``detect`` with a binary endocrine mask and the segmentation labels.  Nearby
fragments are merged to tolerate irregular, lobulated islets:

.. code-block:: python

   from isletscope.islet_detection import IsletDetector
   detector = IsletDetector(min_islet_area=500, min_cell_count=10, merge_distance=8)
   res = detector.detect(endocrine_mask=markers['insulin'], cell_labels=labels, cd3_mask=markers.get('CD3'))
   islet_mask = res['islet_mask']
   single_mask = res['single_mask']


Radial analysis
---------------

Once islets have been identified, radial analysis quantifies spatial
relationships by computing distances from islet centroids.  Use
:class:`isletscope.radial_analysis.RadialAnalyzer` to compute signed
distance maps and bin them.  The ``distance_strategy`` can be switched to
``'boundary'`` for non‑convex islets, and overlap maps highlight where
adjacent islet shells intersect:

.. code-block:: python

   from isletscope.radial_analysis import RadialAnalyzer
   analyzer = RadialAnalyzer(bin_size=20, distance_strategy="boundary")
   dist_map, centroid = analyzer.compute_distance_map(islet_mask)
   bins = analyzer.radial_bins(dist_map)
   hist = analyzer.summarize(bins, cell_mask=markers['CD3'])
   overlap = analyzer.shell_overlap_map(res['islet_labels'])


Tissue subclassification
------------------------

To distinguish different tissue components, call
:class:`isletscope.tissue_classification.TissueClassifier`.  It uses k‑means
clustering in LAB color space plus bright lumen heuristics to assign pixels to
parenchyma, connective tissue, ducts/vessels, background and other classes:

.. code-block:: python

   from isletscope.tissue_classification import TissueClassifier
   classifier = TissueClassifier(n_clusters=4)
   masks = classifier.classify(img_norm)
   parenchyma_mask = masks['parenchyma']


3D anatomical inference
-----------------------

To approximate where a section lies along the head–body–tail axis of the
pancreas and its depth within the organ, use
:class:`isletscope.spatial_inference.SpatialInferer`.  This is a
lightweight heuristic that returns normalized coordinates using vessel
density, endocrine density and tissue composition:

.. code-block:: python

   from isletscope.spatial_inference import SpatialInferer
   inferer = SpatialInferer()
   coords = inferer.infer(img_norm, vessel_mask=masks['ducts_vessels'], islet_mask=islet_mask, tissue_masks=masks)
   print(coords)
