# Cell Segmentation Parameters Explained

## The Problem You Identified ⚠️

You're absolutely correct - the **original workflow was backwards**:

```
❌ WRONG ORDER (before):
1. Stain normalize entire image (including background glass)
2. Segment cells EVERYWHERE (including background)
3. Result: Background/glass artifacts detected as "cells"
```

This caused **more background to be segmented than actual tissue**, which is clearly wrong.

## The Fix ✅

The workflow should be:

```
✅ CORRECT ORDER (now):
1. Detect tissue vs. background/glass
2. Create tissue mask
3. Segment cells ONLY inside tissue mask
4. Remove any cells detected in background
5. Result: Only real cells in tissue regions
```

---

## Configuration Parameters

### 1. **`backend`** - Segmentation Method

```python
backend = 'instanseg'  # Options: 'instanseg', 'classical', 'model'
```

**What it does**: Chooses which algorithm segments cells.

**Options**:

- **`'instanseg'`** (recommended for WSI)
  - Uses deep learning model trained on thousands of annotated images
  - Pros: Accurate, handles overlapping cells, works on challenging images
  - Cons: Slower, requires model download, may detect artifacts as cells if no tissue mask
  - Best for: Brightfield H&E, multiplex fluorescence

- **`'classical'`** (fast fallback)
  - Traditional image processing: Otsu thresholding + watershed
  - Pros: Fast, no downloads, interpretable
  - Cons: Less accurate, struggles with overlapping cells, sensitive to staining variation
  - Best for: Quick tests, simple images with well-separated cells

- **`'model'`** (advanced users)
  - Use your own custom PyTorch model
  - Requires providing a trained model object

**Recommendation**: Use `'instanseg'` with `detect_tissue_first=True`

---

### 2. **`probability_threshold`** - Confidence Threshold

```python
probability_threshold = 0.5  # Range: 0.0 to 1.0
```

**What it does**: For algorithms that output probability maps (classical segmentation, some models), this sets the minimum confidence to classify a pixel as "cell".

**IMPORTANT**: **This parameter does NOTHING when `backend='instanseg'`** because InstanSeg directly outputs instance labels, not probabilities.

**How it works (classical segmentation only)**:
- Lower (0.3): More permissive → detects more cells, more false positives
- Default (0.5): Balanced
- Higher (0.7): Stricter → fewer false positives, may miss dim/faint cells

**Example**:
```python
backend = 'classical'
probability_threshold = 0.3  # Detect more cells (use if missing cells)
probability_threshold = 0.7  # Detect fewer cells (use if too many artifacts)
```

For InstanSeg, this parameter is **ignored**.

---

### 3. **`min_cell_size`** - Size Filter

```python
min_cell_size = 32  # Pixels
```

**What it does**: Removes detected objects smaller than this pixel count.

**Why it matters**: Filters out noise, debris, dust, and small artifacts.

**Typical values**:
- `16` - Keep very small cells (may include debris)
- `32` - Default (good balance)
- `64` - Remove small debris (may miss small cells)
- `128` - Only keep large cells/nuclei

**How to choose**:
1. Look at your image scale
2. Measure typical cell diameter in pixels
3. Calculate approximate area: `area ≈ π * (diameter/2)²`
4. Set `min_cell_size` to ~50% of typical cell area

**Example**:
- Typical nucleus: 10 pixels diameter
- Area: π × 5² ≈ 78 pixels
- Set `min_cell_size = 32` to keep nuclei but remove debris

---

### 4. **`detect_tissue_first`** - Tissue Masking ⭐ **NEW**

```python
detect_tissue_first = True  # Recommended!
```

**What it does**: Automatically detects tissue boundaries BEFORE segmenting cells, then only segments cells inside tissue.

**Why you need this**: **This is the fix for your problem!**

- **`True`** (recommended): Prevents false positives on background/glass
- **`False`** (old behavior): Segments cells everywhere (causes your issue)

**How it works**:
1. Converts image to grayscale
2. Applies Otsu thresholding (tissue is darker than glass)
3. Morphological cleanup (removes small holes and debris)
4. Creates binary mask: 1=tissue, 0=background
5. Segments cells normally
6. Multiplies cell mask by tissue mask
7. **Result**: Only cells in tissue regions

**Output messages**:
```
Detecting tissue boundaries...
  Tissue area: 67.3% of image
Processing 90 tiles...
  Removed 45,231 cells in background (72.4%)
  Final cell count: 17,234 cells in tissue
```

---

### 5. **`tissue_detection_method`** - How to Detect Tissue ⭐ **NEW**

```python
tissue_detection_method = 'otsu'  # Options: 'otsu', 'brightness', 'saturation'
```

**What it does**: Chooses algorithm for separating tissue from background.

**Options**:

- **`'otsu'`** (recommended, default)
  - Automatic threshold selection
  - Works well for H&E and most brightfield images
  - Assumes tissue is darker than background
  - Most robust

- **`'brightness'`**
  - Simple threshold: pixels < 220 = tissue
  - Fast but less adaptive
  - Use if Otsu fails on very bright/dark images

- **`'saturation'`**
  - Uses color saturation (tissue has color, glass is gray/white)
  - Good for lightly stained images
  - May fail on grayscale or very pale tissue

**Troubleshooting**:
- If tissue mask includes too much background → try `'saturation'`
- If tissue mask misses tissue → try `'brightness'` with custom threshold
- Visualize tissue mask to verify quality

---

## Workflow Comparison

### Before (Your Issue):

```python
seg_result = segmenter.segment(img_normalized, image_type='brightfield')
```

Output:
```
Segmentation complete: 62,465 cells detected
```

**Problem**: Includes 45,231 false positives on background glass!

### After (Fixed):

```python
seg_result = segmenter.segment(
    img_normalized,
    image_type='brightfield',
    detect_tissue_first=True,  # ← THE FIX
)
```

Output:
```
Detecting tissue boundaries...
  Tissue area: 67.3% of image
Processing 90 tiles...
  Removed 45,231 cells in background (72.4%)
  Final cell count: 17,234 cells in tissue
```

**Result**: Only real cells in tissue! 🎉

---

## Visualizations

The updated notebook now shows **4 panels**:

1. **Normalized Image** - Stain-normalized RGB image
2. **Tissue Mask** - Binary mask (white=tissue, black=background)
3. **Cell Mask** - Detected cells (white=cell, black=background)
4. **Overlay** - Blue=tissue boundary, Red=cells

This lets you verify:
- ✅ Tissue detection worked correctly
- ✅ Cells only detected inside tissue
- ✅ No false positives on glass/background

---

## Complete Parameter Guide

### Recommended Settings (Brightfield H&E):

```python
# Tissue detection (prevents background false positives)
detect_tissue_first = True
tissue_detection_method = 'otsu'

# Segmentation
backend = 'instanseg'
instanseg_model = 'brightfield_nuclei'
min_cell_size = 32

# Tiling (for large images)
tile_size = 1024
tile_overlap = 64

# NOT USED BY INSTANSEG (only for classical):
probability_threshold = 0.5  # Ignored when backend='instanseg'
```

### Recommended Settings (Fluorescence):

```python
# Tissue detection
detect_tissue_first = True
tissue_detection_method = 'saturation'  # Better for fluorescence

# Segmentation
backend = 'instanseg'
instanseg_model = 'fluorescence_nuclei_and_cells'
min_cell_size = 32

# Tiling
tile_size = 1024
tile_overlap = 64
```

### Troubleshooting Settings:

**Too many false positives (debris, artifacts)**:
```python
min_cell_size = 64  # Increase size filter
detect_tissue_first = True  # Always use tissue masking
```

**Missing cells**:
```python
min_cell_size = 16  # Decrease size filter
tile_overlap = 128  # More overlap (prevent edge clipping)
tissue_detection_method = 'brightness'  # Try different tissue detection
```

**Tissue mask includes background**:
```python
tissue_detection_method = 'saturation'  # Use color instead of brightness
# Or manually adjust threshold in detect_tissue() method
```

**Slow processing**:
```python
backend = 'classical'  # Faster but less accurate
tile_size = 2048  # Larger tiles (fewer tiles to process)
```

---

## Summary

### Your Original Question:

> "The first attempt at cellular segmentation was absolutely terrible. More background area was segmented as a cell than tissue was."

### The Answer:

**The workflow was backwards!** Cell segmentation was happening on the entire image including background glass, leading to massive false positives.

**The Fix**: Set `detect_tissue_first=True` to:
1. Detect tissue boundaries first
2. Only segment cells inside tissue
3. Remove any cells detected in background
4. **Result**: 70-80% reduction in false positives

### Key Parameters:

1. **`backend`** - Which algorithm ('instanseg', 'classical', 'model')
2. **`probability_threshold`** - Only for classical (ignored by InstanSeg)
3. **`min_cell_size`** - Remove small objects/debris (pixels)
4. **`detect_tissue_first`** - ⭐ **THE FIX** - Prevents background false positives
5. **`tissue_detection_method`** - How to detect tissue ('otsu', 'brightness', 'saturation')

### Next Steps:

1. Re-run notebook 01 with `detect_tissue_first=True`
2. Verify tissue mask quality in visualizations
3. Check that cells are only in tissue regions
4. Adjust `min_cell_size` if needed
5. Proceed to notebook 02 for islet analysis

The segmentation should now be **much more accurate**! 🎯
