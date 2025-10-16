# Crop Identification Using Satellite Imagery for Northern India

## Hackathon Link: [https://cni.iisc.ac.in/hackathons/gen-AI-2025/]

This project is a **Top 5 solution** (from 3000+ participants, selected as top 100) in the **IndiaAI Impact Gen-AI Hackathon** organized by Kaggle and IndiaAI. It develops a deep learning-based crop classification model for fragmented farmlands in Northern India using multi-spectral Sentinel-2 satellite imagery. The model addresses challenges like diverse crop types, small farm sizes, and class imbalance by leveraging advanced spectral analysis and custom loss functions.

### Problem Statement
- **Objective**: Classify 6 major crop types (Gram, Maize, Mustard, Sugarcane, Wheat, OtherCrop) from satellite images.
- **Dataset**: Sentinel-2 imagery with 12 bands, covering ~1,173 training/validation samples (928 train, 119 val, 126 test).
- **Challenges**: Severe class imbalance (e.g., Wheat: 40% pixels vs. Gram: 1.7%), spectral variability, and no-data regions.
- **Impact**: Enables precision agriculture, yield prediction, and sustainable farming in fragmented landscapes.

### Key Innovations
- **Enhanced Dataset**: Processed 12-band inputs into **34-band rasters** by retaining original bands and computing **22 vegetation/spectral indices** (e.g., NDVI, EVI, SAVI, NDMI, GVMI, NDRE1/2, NDYI, PSRI, GNDVI, REP, OSAVI, WDRVI, MCARI, TCARI, CCCI, MTCI, CIre, SIPI, BSI, EXG, RDVI).
- **Custom Loss**: **Focal Tversky Loss** with class weights ([0.40, 0.15, 0.05, 0.35, 0.03, 0.04]) to handle imbalance and focus on hard examples (α=0.3, β=0.7, γ=4/3).
- **Custom Neck Module**: **CropVIAttentionNeck** – A vegetation index (VI) attention mechanism tailored for crop-specific band importance, integrated with **TerraTorch** and **Prithvi** backbone.
- **Training Enhancements**: Learning rate scheduler (CosineAnnealingWarmRestarts), mixed precision (16-bit), and gradient accumulation for efficient fine-tuning.
- **Evaluation**: Multiclass Jaccard Index (mIoU) monitored; best val mIoU: ~0.45 (Epoch 56).

The model achieves robust performance on imbalanced data, with predictions formatted as Run-Length Encoded (RLE) strings for submission.

## Dataset Preparation

1. **Input**: 12-band Sentinel-2 TIFFs (bands: B01, Blue, Green, Red, B05, B06, B07, NIR_Narrow, B8A, B09, SWIR1, SWIR2).
2. **Processing**:
   - Compute 22 spectral indices using Rasterio and NumPy.
   - Stack with original 12 bands → 34-band outputs.
   - Normalize using pre-computed means/stds (e.g., NDVI mean: 0.213, std: 0.111).
   - Handle no-data: Replace with 0 (inputs), -1 (labels).
3. **Splits**: Train (928), Val (119), Test (126) – Defined via `.txt` files.
4. **Augmentations**: D4 rotations + ToTensorV2 (via Albumentations).

**Code Snippet** (from `code.ipynb`):
```python
# Spectral indices calculation (e.g., NDVI, EVI, etc.)
def calculate_spectral_indices(src):
    # ... (reads bands, computes 22 indices with epsilon for div-by-zero)
    return np.array(indices_cleaned, dtype=np.float32)

# Batch processing for train/val/test
process_directory(input_dir, output_dir, band_indices_to_keep=[1,2,3,4,5,6,7,8,9,10,11,12])
```

## Model Architecture

- **Backbone**: Prithvi (pre-trained on Earth observation data via TerraTorch).
- **Neck**: Custom CropVIAttentionNeck – Applies class-specific attention weights to VI bands (e.g., high weight on NDVI/EVI for Gram/Maize).
- **Head**: Semantic segmentation head with dropout (0.25).
- **Input**: 34 channels × 256×256 patches.
- **Output**: 6-class segmentation map.

**Custom Neck Registration:**
```python
@TERRATORCH_NECK_REGISTRY.register()
class CropVIAttentionNeck(nn.Module):
    def __init__(self, in_channels, num_classes=6):
        # ... (attention weights per crop, e.g., Gram: NDVI=0.9, EVI=0.8)
        # Forward: Weighted fusion of multi-scale features
```
## Training

- **Framework**: PyTorch Lightning + TerraTorch.
- **Hyperparameters**:

  - Batch Size: 16 (accumulated: 3 → effective 48).
  - LR: 1e-5, Weight Decay: 0.01.
  - Epochs: 120, Scheduler: CosineAnnealingWarmRestarts (T_0=10).
  - Optimizer: AdamW.
  - Precision: 16-mixed.


- **Monitoring**: TensorBoard logger, ModelCheckpoint on val mIoU.
- **Custom Task**: CustomSegmentationTaskWithEQLv2 overrides loss and logs LR.

**Training Snippet:**
```python
model = CustomSegmentationTaskWithEQLv2(
    model_name="prithvi",
    model_args={"neck": "CropVIAttentionNeck", "num_classes": 6},
    loss_args={"criterion": FocalTverskyLoss(...)},
    # ...
)

trainer = pl.Trainer(..., max_epochs=120, callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=datamodule)

```

###Training Logs (Excerpt):

- **Epoch 0**: Val mIoU = 0.00
- **Epoch 26**: Val mIoU = 0.371
- **Epoch 52**: Val mIoU = 0.420 (Best: 0.449 at Epoch 56)

## Inference & Submission

- **Prediction**: Load best checkpoint, predict on test inputs → Save 6-class TIFFs.
- **Post-Processing**: Apply masks, compute RLE for valid pixels (where mask==1).
- **Output: submission**.csv with id (e.g., filename_0) and label (RLE string).

**Inference Snippet:**
```python
preds = trainer.predict(model, datamodule=datamodule, ckpt_path=best_ckpt_path)
# ... (save TIFFs, compute RLE, generate CSV)
df.to_csv("submission.csv", index=False)
```

## 📊 Model Performance Metrics

| **Metric** | **Train** | **Val** | **Test** |
|-------------|-----------|----------|-----------|
| **Multiclass Jaccard Index (mIoU)** | 0.52 | 0.45 | 0.43 |
| **Per-Class IoU (Wheat / Maize / Gram)** | 0.68 / 0.55 / 0.32 | 0.62 / 0.48 / 0.28 | 0.60 / 0.46 / 0.25 |

## Dependencies
**Run in Kaggle (GPU: T4) or local environment:**
```python
# Core
pip install torch torchvision torchaudio lightning albumentations rasterio numpy pandas

# TerraTorch (custom install via notebook)
# !pip install terratorch -q  # See notebook for sys.path append

# Others: terratorch, pretrainedmodels, etc. (pre-installed in Kaggle)
```


