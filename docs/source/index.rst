.. IsletScope documentation master file
   This file contains the high level outline of the documentation.

Welcome to **IsletScope's** documentation!
==========================================

IsletScope is a modular framework for pancreatic histology analysis.  It provides tools for stain normalization, cell segmentation, islet detection, radial analysis, tissue classification and 3D inference.  Each component is available as a Python class with a simple, well‑documented interface.

.. image:: _static/logo.png
   :align: center
   :alt: IsletScope logo

Environment
-----------

Create the environment directly from the YAML that matches your hardware.  Use ``environment.yml`` for GPU systems (Blackwell B200, RTX 6000 Ada) and ``environment.cpu.yml`` for CPU-only setups.  Activate with ``mamba activate isletscope`` (or ``isletscope-cpu``) and install the package editable via ``pip install -e .``.  CuPy is optional; if it is available on your channels and you want GPU-accelerated stain normalization, install it with ``mamba install -n isletscope cupy-cuda12x`` after creating the environment.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   tutorial
   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
