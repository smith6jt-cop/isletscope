# IsletScope

IsletScope is a comprehensive analysis framework for pancreatic histology and microscopy images. It combines robust stain normalization, instance-level cell segmentation, flexible islet detection, radial distance mapping and three‑dimensional inference into a single, cohesive toolset. Both brightfield and multiplex fluorescence workflows are supported.

The primary goal of IsletScope is to provide a reproducible and scalable workflow for analyzing both brightfield and multiplex fluorescent images of pancreas tissue.  It is designed for users who require fine control over each processing step while taking advantage of GPU acceleration and parallel processing on a variety of hardware architectures (including Blackwell and Ada GPUs).

## Features

* **Automated stain normalization** – blind Macenko/Vahadane estimation with optional GPU acceleration; percentile scaling for fluorescence.
* **Instance-level cell segmentation** – integrate with deep‑learning models such as InstanSeg (auto-detected when installed) and a GPU-friendly classical fallback for brightfield/fluorescence.
* **Islet detection and classification** – merge nearby endocrine fragments, tolerate irregular shapes, and distinguish endocrine clusters versus single endocrine cells with optional CD3 quantification.
* **Radial analysis** – compute concentric radial distance maps (centroid or boundary based) and overlap fields to approximate lobular organization.
* **Tissue subclassification** – distinguish parenchyma, connective tissue, ducts/vessels, background and other structures using LAB clustering plus lumen heuristics.
* **3D anatomical inference** – map 2D sections into a 3D context based on vascular density, endocrine density, tissue composition and known head–body–tail gradients.
* **Notebook‑based workflow** – run each stage interactively in Jupyter notebooks with embedded visualizations and tests.

## Quick Start

**📖 See [QUICKSTART.md](QUICKSTART.md) for detailed setup and usage instructions.**

The environment is already installed! Run notebooks with:
```bash
./run_jupyter.sh
```

### Notebook Workflow

| Notebook | Purpose |
|---|---|
| `00_train_instanseg.ipynb` | *(Optional)* Train custom InstanSeg models on annotated data. |
| `01_stain_normalization.ipynb` | **⭐ Start here:** Stain normalization and cell segmentation with InstanSeg. |
| `02_segmentation.ipynb` | Islet detection and radial distance analysis. |
| `03_islet_detection.ipynb` | Tissue classification and 3D spatial inference. |

## Installation (mamba)

Choose the YAML that matches your hardware and create the environment directly with mamba. CuPy is optional; install later only if your channels provide it.

GPU (B200 Blackwell / RTX 6000 Ada, CUDA 12.x):
```bash
mamba env create -f environment.yml
mamba activate isletscope
pip install -e .
# Optional CuPy if available on your channels:
# mamba install -n isletscope cupy-cuda12x
```

CPU-only:
```bash
mamba env create -f environment.cpu.yml
mamba activate isletscope-cpu
pip install -e .
```

## Documentation

Full documentation is available on Read the Docs.  It includes a tutorial, API reference and design notes.  See the `docs/` directory for the source.

### Notebooks

The `notebooks/` directory contains six stepwise notebooks:

1. `01_stain_normalization.ipynb` – blind Macenko/Vahadane estimation (brightfield) or percentile scaling (fluorescence).
2. `02_segmentation.ipynb` – InstanSeg/custom model integration and classical fallback with marker extraction helpers.
3. `03_islet_detection.ipynb` – flexible islet detection with fragment merging and CD3 quantification.
4. `04_radial_analysis.ipynb` – centroid/boundary radial shells plus overlap maps.
5. `05_tissue_subclassification.ipynb` – parenchyma/connective/duct-vessel/background masks.
6. `06_3D_inference.ipynb` – heuristic head–body–tail and depth inference using vessels, islets and tissue masks.
