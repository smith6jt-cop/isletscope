# IsletScope

IsletScope is a comprehensive analysis framework for pancreatic histology and microscopy images. It combines robust stain normalization, instance-level cell segmentation, flexible islet detection, radial distance mapping and three‑dimensional inference into a single, cohesive toolset.

The primary goal of IsletScope is to provide a reproducible and scalable workflow for analyzing both brightfield and multiplex fluorescent images of pancreas tissue.  It is designed for users who require fine control over each processing step while taking advantage of GPU acceleration and parallel processing on a variety of hardware architectures (including Blackwell and Ada GPUs).

## Features

* **Automated stain normalization** – estimate and normalize staining vectors across heterogeneous images without manual tuning.
* **Instance-level cell segmentation** – integrate with deep‑learning models such as InstanSeg and fallback classical methods for GPU‑accelerated segmentation.
* **Islet detection and classification** – identify endocrine clusters versus single endocrine cells using flexible rules that tolerate irregular shapes.
* **Radial analysis** – compute concentric radial distance maps from islet centroids to explore spatial relationships with other cell types.
* **Tissue subclassification** – distinguish parenchyma, connective tissue, ducts, vessels, nerves and other structures using multi‑class segmentation.
* **3D anatomical inference** – map 2D sections into a 3D context based on vascular and ductal landmarks and known head–body–tail anatomy.
* **Notebook‑based workflow** – run each stage interactively in Jupyter notebooks with embedded visualizations and tests.

## Quick Start

Follow the tutorial notebooks in the `notebooks/` directory for a step‑by‑step walkthrough.  Each notebook corresponds to one stage of the pipeline:

| Notebook | Purpose |
|---|---|
| `01_stain_normalization.ipynb` | Estimate and apply stain normalization. |
| `02_segmentation.ipynb` | Perform cell segmentation using InstanSeg or a placeholder classical method. |
| `03_islet_detection.ipynb` | Detect islets and endocrine cells based on segmentation results. |
| `04_radial_analysis.ipynb` | Compute radial distance maps from detected islets. |
| `05_tissue_subclassification.ipynb` | Segment tissue into parenchyma, connective tissue and other classes. |
| `06_3D_inference.ipynb` | Place 2D sections into a 3D anatomical context. |

## Installation

IsletScope is distributed as a Python package.  You can install it in a virtual environment with:

```bash
pip install -e .[all]
```

This will install the core dependencies along with optional packages for GPU acceleration and documentation.

## Documentation

Full documentation is available on Read the Docs.  It includes a tutorial, API reference and design notes.  See the `docs/` directory for the source.
