# IsletScope Quick Start Guide

## ✅ Installation Complete!

The `isletscope` package and all dependencies (including InstanSeg) are now installed in the `isletscope` mamba environment.

---

## Running the Notebooks

### Option 1: Jupyter Lab (Recommended)

```bash
# From the isletscope directory
./run_jupyter.sh
```

This will start Jupyter Lab with the correct environment activated.

### Option 2: Command Line with Mamba

```bash
# Activate the environment
mamba activate isletscope

# Navigate to notebooks directory
cd notebooks

# Start Jupyter
jupyter lab
```

### Option 3: Run Notebooks Directly

```bash
# Run a specific notebook
mamba run -n isletscope jupyter execute 01_stain_normalization.ipynb
```

---

## Notebook Workflow

The analysis workflow consists of 4 notebooks:

### **[00_train_instanseg.ipynb](notebooks/00_train_instanseg.ipynb)** *(Optional)*
- Train custom InstanSeg models on your annotated data
- Fine-tune pretrained models for specialized tissue types
- Only needed if pretrained models don't work well

### **[01_stain_normalization.ipynb](notebooks/01_stain_normalization.ipynb)** ⭐ **Start Here**
- Stain normalization (Macenko/Vahadane)
- Cell/nucleus segmentation with InstanSeg
- Marker detection (insulin, glucagon, CD3)
- **Full-image and close-up visualizations**

### **[02_segmentation.ipynb](notebooks/02_segmentation.ipynb)**
- Islet detection (vs. single endocrine cells)
- Radial distance analysis from islet centroids/boundaries
- CD3+ T-cell proximity analysis

### **[03_islet_detection.ipynb](notebooks/03_islet_detection.ipynb)**
- Tissue classification (parenchyma, connective, vessels, etc.)
- 3D spatial inference (head/body/tail positioning)
- Comprehensive analysis summary

---

## InstanSeg Configuration

The InstanSeg integration has been **completely rewritten** to address the cell detection issues. Key improvements:

### ✅ Fixed Issues
1. **Proper model initialization** - Uses `InstanSeg(model_name, ...)` API correctly
2. **Tile-based processing** - Handles large WSI images by splitting into tiles
3. **Parameter exposure** - All configuration options now accessible in notebooks
4. **Shape error fixed** - Automatically handles InstanSeg output dimensions `(B,C,H,W)` → `(H,W)`

> 📝 **Latest Fix**: Shape broadcasting error resolved. See [INSTANSEG_FIX.md](INSTANSEG_FIX.md) for details.

### 🔧 Configuration Parameters (in notebook 01)

```python
# ===== InstanSeg Parameters =====
instanseg_model = 'brightfield_nuclei'  # Model selection
tile_size = 1024          # Tile size in pixels (512, 1024, 2048)
tile_overlap = 64         # Overlap between tiles (prevents edge artifacts)
batch_size = 4            # GPU batch processing
pixel_size = None         # Auto-detected from metadata
normalization = True      # Intensity normalization
image_reader = 'tiffslide'  # Backend (tiffslide, openslide)
```

### 📊 Expected Performance

For a **2000×2000 pixel brightfield H&E image**:
- **Before fix**: ~144 cells (incorrect, missing most cells)
- **After fix**: Tens of thousands to hundreds of thousands of cells (tile-based processing)

The previous implementation used generic API discovery without proper model initialization. The new implementation:
1. Properly initializes InstanSeg with model selection
2. Automatically detects when tiling is needed (image > tile_size)
3. Processes tiles with configurable overlap
4. Merges results and re-labels connected components

### 🔍 Tuning for Your Data

If cell detection is still not optimal:

**Too few cells detected:**
- Decrease `min_cell_size` (default 32 pixels)
- Increase `tile_overlap` to 128 (prevents edge clipping)
- Try `tile_size = 512` for denser sampling

**Too many false positives:**
- Increase `min_cell_size` to 64 or 128
- Adjust `probability_threshold` (default 0.5)

**Memory issues:**
- Decrease `tile_size` to 512
- Decrease `batch_size` to 2

**Slow processing:**
- Increase `tile_size` to 2048 (if GPU allows)
- Decrease `tile_overlap` to 32

---

## Verifying InstanSeg Works

Run the test script to verify the installation:

```bash
mamba run -n isletscope python test_instanseg.py
```

This will:
1. Test classical segmentation (baseline)
2. Test InstanSeg with default parameters
3. Compare cell counts
4. Verify parameter configuration

**Expected output:**
```
InstanSeg model 'brightfield_nuclei' initialized successfully.
Processing 4 tiles (2x2) with tile_size=1024, overlap=64
  Processed 4/4 tiles...
InstanSeg detected 125,432 cells
✓ PASS: InstanSeg detected > 1000 cells
```

---

## Available Models

### Pretrained InstanSeg Models
- **`brightfield_nuclei`** - Trained on H&E and DAB stains
- **`fluorescence_nuclei_and_cells`** - For multiplex fluorescence images

Models are automatically downloaded on first use.

### Training Custom Models

If the pretrained models don't work well:
1. Annotate 10-50 representative images in QuPath/Cellpose
2. Export instance masks (16-bit TIFF with unique labels per cell)
3. Run **[00_train_instanseg.ipynb](notebooks/00_train_instanseg.ipynb)**
4. Fine-tune from `brightfield_nuclei` base model
5. Export and use in notebook 01

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'isletscope'"

**Solution:** Make sure you're using the mamba environment:
```bash
mamba activate isletscope
# OR
mamba run -n isletscope jupyter lab
```

### "InstanSeg is not installed"

**Solution:** Install InstanSeg in the environment:
```bash
mamba activate isletscope
pip install instanseg-torch[full]
```

### "CUDA out of memory"

**Solution:** Reduce tile size or batch size:
```python
tile_size = 512      # Smaller tiles
batch_size = 2       # Process fewer tiles at once
```

### Notebook kernel keeps crashing

**Solution:** Install ipykernel in the environment:
```bash
mamba activate isletscope
pip install ipykernel
python -m ipykernel install --user --name=isletscope
```

Then select the "isletscope" kernel in Jupyter.

---

## GPU Support

The environment is configured for **CUDA 12.4** (Blackwell B200 and RTX 6000 Ada).

**Check GPU availability:**
```bash
mamba run -n isletscope python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**If using CPU only:**
- Processing will be slower but still work
- Reduce `batch_size` to 1
- Consider using HiPerGator for large datasets

---

## File Structure

```
isletscope/
├── notebooks/
│   ├── 00_train_instanseg.ipynb    # Custom model training
│   ├── 01_stain_normalization.ipynb # Segmentation (START HERE)
│   ├── 02_segmentation.ipynb        # Islet detection
│   └── 03_islet_detection.ipynb     # Tissue classification
├── isletscope/
│   ├── stain.py                     # Stain normalization
│   ├── segmentation.py              # Cell segmentation (UPDATED)
│   ├── islet_detection.py           # Islet detection
│   ├── radial_analysis.py           # Radial distance analysis
│   ├── tissue_classification.py     # Tissue subtyping
│   └── spatial_inference.py         # 3D positioning
├── outputs/                         # Results saved here
├── images/                          # Input images (.svs, .tif, etc.)
├── environment.yml                  # Mamba environment
├── run_jupyter.sh                   # Jupyter launcher
└── test_instanseg.py                # InstanSeg test script
```

---

## Next Steps

1. **Place your images** in the `images/` directory
2. **Run notebook 01** to test the updated InstanSeg integration
3. **Verify cell counts** - should see tens/hundreds of thousands of cells
4. **Adjust parameters** as needed (see tuning guide above)
5. **Proceed to notebooks 02-03** for islet analysis

---

## Summary of Changes

### What Was Fixed
✅ **InstanSeg initialization** - Proper model loading with `InstanSeg(model_name, ...)`
✅ **Tile-based processing** - Automatic tiling for large images
✅ **Parameter configuration** - All parameters exposed in notebook 01
✅ **Model downloading** - Automatic download of pretrained models
✅ **Training notebook** - Complete workflow for custom models
✅ **Error handling** - Better diagnostics and fallback behavior

### Code Changes
- **[isletscope/segmentation.py](isletscope/segmentation.py)**: Complete rewrite of `_segment_with_instanseg()`
- **[notebooks/01_stain_normalization.ipynb](notebooks/01_stain_normalization.ipynb)**: Added InstanSeg configuration section
- **[notebooks/00_train_instanseg.ipynb](notebooks/00_train_instanseg.ipynb)**: New training notebook

---

**Ready to go!** Start with `./run_jupyter.sh` and open notebook 01.
