Tutorial
========

This tutorial provides a high‑level overview of the IsletScope workflow.  For
practical examples see the Jupyter notebooks in the repository under
``notebooks/``.  The workflow is divided into six stages.

Stain normalization
-------------------

Images acquired on different scanners or at different times often exhibit
variation in staining intensity and hue.  The :class:`isletscope.stain.StainNormalizer`
implements an automated version of the Macenko algorithm to estimate the
underlying stain vectors and normalize images to a common basis.

.. code-block:: python

   from isletscope.stain import StainNormalizer
   import cv2

   img = cv2.imread('sample.tif')
   normalizer = StainNormalizer()
   normalizer.estimate_stain_matrix(img)
   img_norm = normalizer.normalize(img)


Cell segmentation
-----------------

Segmenting individual cells is the foundation for downstream analysis.  Use
:class:`isletscope.segmentation.CellSegmenter` to perform segmentation via
a deep‑learning model or a classical thresholding fallback:

.. code-block:: python

   from isletscope.segmentation import CellSegmenter
   segmenter = CellSegmenter()
   result = segmenter.segment(img_norm)
   mask = result['mask']
   labels = result['labels']


Islet detection
---------------

Islet detection classifies endocrine clusters into islets or single cells.
Instantiate :class:`isletscope.islet_detection.IsletDetector` and call
``detect`` with a binary endocrine mask and the segmentation labels:

.. code-block:: python

   from isletscope.islet_detection import IsletDetector
   detector = IsletDetector(min_islet_area=500, min_cell_count=10)
   res = detector.detect(endocrine_mask, labels)
   islet_mask = res['islet_mask']
   single_mask = res['single_mask']


Radial analysis
---------------

Once islets have been identified, radial analysis quantifies spatial
relationships by computing distances from islet centroids.  Use
:class:`isletscope.radial_analysis.RadialAnalyzer` to compute signed
distance maps and bin them:

.. code-block:: python

   from isletscope.radial_analysis import RadialAnalyzer
   analyzer = RadialAnalyzer(bin_size=20)
   dist_map, centroid = analyzer.compute_distance_map(islet_mask)
   bins = analyzer.radial_bins(dist_map)
   hist = analyzer.summarize(bins, cell_mask=endocrine_mask)


Tissue subclassification
-----------------------

To distinguish different tissue components, call
:class:`isletscope.tissue_classification.TissueClassifier`.  It uses k‑means
clustering in LAB color space to assign pixels to parenchymal, connective
and other classes:

.. code-block:: python

   from isletscope.tissue_classification import TissueClassifier
   classifier = TissueClassifier(n_clusters=3)
   masks = classifier.classify(img_norm)
   parenchyma_mask = masks['parenchyma']


3D anatomical inference
----------------------

To approximate where a section lies along the head–body–tail axis of the
pancreas and its depth within the organ, use
:class:`isletscope.spatial_inference.SpatialInferer`.  This is a
placeholder implementation that returns normalized coordinates based on
simple heuristics:

.. code-block:: python

   from isletscope.spatial_inference import SpatialInferer
   inferer = SpatialInferer()
   coords = inferer.infer(img_norm)
   print(coords)
