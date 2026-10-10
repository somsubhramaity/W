---
title: Factory Safety Monitor
emoji: 🦺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# Factory Safety Monitor

Real-time factory monitoring prototype for PPE compliance, smoke, and fire alerts. It can consume an uploaded video, a webcam, or an RTSP stream and records responder-ready snapshots with camera and zone context.

## Architecture

```text
Video / webcam / RTSP
     |
     v
  Streamlit dashboard ----> annotated live feed + compliance and alert summaries
     |
     v
  pipeline.py
    |                         |
    | PPE YOLO                | fire_smoke.py
    | models/ppe_yolov8.pt    | Hugging Face YOLOv8n checkpoint
    +------------+------------+
           |
           v
  required-gear checks + camera/zone/timestamp context
           |
           v
  alerts.py --> outputs/alerts/ snapshots, alerts.csv, alerts.json
           +--> optional Telegram photo notification
```

The PPE model uses the classes `helmet`, `gloves`, `vest`, `boots`, `goggles`, `none`, `Person`, `no_helmet`, `no_goggle`, `no_gloves`, and `no_boots`. Ultralytics ByteTrack runs persistently on every frame, and each `Person` receives its track ID. Fire/smoke classes are `0=smoke`, `1=fire`. Gear requirements are configurable; the default requires a helmet and vest. Gear boxes are associated with a person when their center falls inside the person box. A missing required box or an explicit `no_*` detection marks that person as a violation.

## Windows Setup

Use Python 3.9 or newer. The repository's `models/ppe_yolov8.pt` PPE checkpoint is required. The smoke/fire checkpoint downloads automatically on first model initialization to `models/fire_smoke/best.pt`, then can be reused offline.

```powershell
cd PPL-detection
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Start the dashboard:

```powershell
streamlit run app.py
```

Choose Upload video, Webcam, or RTSP stream in the sidebar. Set a camera ID and zone, adjust required gear, and start monitoring. Webcam/RTSP capture depends on local OpenCV access and stream/network availability.

Run the recorded-video demo:

```powershell
python demo.py --source "path\to\factory-footage.mp4" --camera-id line-1 --zone "Assembly Line"
```

The annotated video is written to `outputs/factory_safety_annotated.mp4` by default. Alert snapshots and logs are written to `outputs/alerts/`, `outputs/alerts.csv`, and `outputs/alerts.json`. At end-of-video, the demo also prints and saves a track-level summary to `outputs/summary.json`, then sends one Telegram text summary followed by up to three cropped helmet-violator snapshots when credentials are configured.

## Configuration

Edit `config.yaml`:

- `models.ppe`: path to the PPE YOLO weights.
- `models.fire_smoke`: local cache path for the Hugging Face `best.pt` weights.
- `inference.confidence`: detection confidence threshold.
- `inference.image_size`: inference size, capped at 640 for CPU-friendly operation.
- `inference.frame_stride`: run fire/smoke detection every Nth video frame; person tracking and PPE detection still run on every frame.
- `tracking.minimum_person_frames`: minimum observed frames before a track can count as a gear violation (default `15`).
- `tracking.missing_gear_ratio`: required fraction of observed frames missing gear (default `0.5`).
- `required_gear`: any combination of `helmet`, `vest`, `gloves`, and `boots`.
- `camera_zones`: mapping of camera IDs to area names.
- `alerts.cooldown_seconds`: same event/camera suppression window.
- `alerts.confirmation_frames`: consecutive inference frames required for fire/smoke alerts.
- `alerts.telegram_live_enabled`: send individual live alerts to Telegram; defaults to `false`, so Telegram receives only the end-of-video summary.

Telegram is optional. Copy `.env.example` to `.env`, then set `TELEGRAM_BOT_TOKEN` and either `TELEGRAM_CHAT_ID` or `TELEGRAM_CHAT_IDS` (the latter accepts comma-separated chat IDs). Without both a token and at least one chat ID, the end summary is still printed, saved, and shown in the dashboard; Telegram delivery is skipped. Live console/dashboard alerts continue to work. Never commit `.env` or credentials.

At the end of an uploaded dashboard video, the same summary appears in the UI with a **Send summary to Telegram** button for manual resend. The summary counts unique ByteTrack person IDs; a track is counted without helmet (or vest) only when it has at least the configured minimum number of observations and is missing that item in at least the configured fraction of those frames. Summary records include first/last frame and video time plus the best-confidence helmet-violation crop per track.

## Limitations And Licensing

This is a hackathon prototype, not a certified life-safety system. The fire/smoke checkpoint is a small YOLOv8n model reported at roughly 0.75 mAP50; that figure has not been independently validated on your factory footage. False positives and missed detections are possible. Requiring consecutive inference frames and applying a per-camera cooldown reduce alert noise but do not establish safety. Person-to-gear association uses bounding-box containment and can be inaccurate with occlusion, distant workers, or overlapping people. Webcam and RTSP behavior should be tested on the target machine/network.

Ultralytics is licensed under AGPL-3.0; review its terms and the checkpoint's Hugging Face model-card license before distribution or deployment. The earlier traditional ML and training documentation below describes the original PPE project and is retained for reference.

---

## Legacy PPE Training Project

This project implements a comprehensive Personal Protective Equipment (PPE) detection system using both deep learning (YOLO) and traditional machine learning approaches.

## 🚀 Quick Start

Get up and running in 3 simple steps - **no training required!** The repository includes pre-trained models.

```bash
# 1. Clone the repository (includes pre-trained models)
git clone https://github.com/tob-git/PPL-detection.git
cd PPL-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the web UI
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and start detecting PPE immediately!

**Note**: The repository includes pre-trained models in the `models/` directory, so you can skip the training step and start using the application right away.

## Project Overview

The project meets all required machine learning pipeline components:

### ✅ I. Dataset with Train/Test Split
- **Dataset**: Construction PPE dataset with 11 classes
- **Training set**: 1,132 images
- **Validation set**: 143 images  
- **Test set**: 141 images
- **Classes**: helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots

### ✅ II. Feature Extraction
Multiple feature types extracted from images:
- **Color Features**: RGB and HSV histograms (160 features)
- **Texture Features**: Sobel gradients, Laplacian variance, gray-level statistics
- **Shape Features**: Contour-based features, edge density
- **Statistical Features**: Mean, std, median, percentiles per channel

### ✅ III. Feature Selection
Three feature selection methods implemented:
- **SelectKBest**: ANOVA F-test based selection
- **PCA**: Principal Component Analysis for dimensionality reduction
- **RFE**: Recursive Feature Elimination with Random Forest

### ✅ IV. Multiple Classifiers
Six different classifiers implemented:
1. **Decision Tree**: Simple tree-based classifier
2. **Random Forest**: Ensemble of decision trees
3. **XGBoost**: Gradient boosting framework
4. **KNN**: K-Nearest Neighbors
5. **SVM**: Support Vector Machine with RBF kernel
6. **ANN (MLP)**: Multi-layer Perceptron neural network

### ✅ V. Performance Evaluation
Comprehensive metrics computed:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted precision
- **Recall**: Per-class and weighted recall
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **Comparative Analysis**: Metrics across all classifier-feature combinations

## Project Structure

```
PPL detection/
├── ppe.py                      # YOLO-based detection pipeline
├── ml_pipeline.py              # Traditional ML classification pipeline ⭐
├── infer.py                    # YOLO inference script
├── inferonnx.py               # ONNX inference script
├── requirements.txt            # Python dependencies
├── construction-ppe.yaml       # Dataset configuration
├── datasets/
│   └── construction-ppe/
│       ├── images/            # Image files
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/            # YOLO format labels
│           ├── train/
│           ├── val/
│           └── test/
├── models/
│   ├── ppe_yolov8.pt          # Trained YOLO model
│   └── ppe_yolov8.onnx        # ONNX format model
└── outputs/
    ├── ml_pipeline/            # ML pipeline results
    │   ├── confusion_matrix_*.png
    │   ├── metrics_comparison.png
    │   ├── detailed_metrics.csv
    │   ├── features_train.pkl
    │   └── features_test.pkl
    ├── label_vis_train/        # Label visualizations
    └── label_vis_val/
```

## Installation & Setup

### 1. Clone the Repository (if applicable)

```bash
git clone https://github.com/tob-git/PPL-detection.git
cd PPL-detection
```

Or navigate to your project directory.

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

The Construction-PPE dataset needs to be downloaded before training. You have two options:

#### Option A: Automatic Download (Recommended)

The dataset will be automatically downloaded when you first run the training script:

```bash
python ppe.py
```

The YOLOv8 training script will:
- Automatically download the dataset from the official source
- Extract it to `datasets/construction-ppe/`
- Organize images into train/val/test splits

#### Option B: Manual Download

If you prefer to download manually:

1. Download the dataset:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
   ```

2. Extract to the correct location:
   ```bash
   unzip construction-ppe.zip -d datasets/
   ```

3. Verify the structure:
   ```
   datasets/construction-ppe/
   ├── images/
   │   ├── train/  (1,132 images)
   │   ├── val/    (143 images)
   │   └── test/   (141 images)
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```

**Dataset Details:**
- **Size**: 178.4 MB
- **Total Images**: 1,416 images
- **Classes**: 11 PPE-related classes
- **Source**: Ultralytics Construction-PPE dataset
- **License**: Check LICENSE file in dataset directory

## Usage

### Option 1: Interactive Web UI (Easiest) 🌐

Launch the Streamlit web interface for easy image upload and real-time predictions:

```bash
streamlit run app.py
```

This will:
- Open a web browser at `http://localhost:8501`
- Provide a user-friendly interface for uploading images
- Allow you to choose between YOLO or Traditional ML models
- Display predictions with visual annotations
- Compare different classifiers side-by-side

**UI Features:**
- ✅ Drag-and-drop image upload
- ✅ Real-time YOLO object detection with bounding boxes
- ✅ Traditional ML classification with 6+ classifiers
- ✅ Confidence scores and safety assessments
- ✅ Color-coded results (green = safe, red = unsafe)

**Note**: You must train models first (see Option 2 or 3) before using the UI.

### Option 2: Traditional ML Pipeline (All Requirements) ⭐

Run the complete ML pipeline with feature extraction, selection, and multiple classifiers:

```bash
python ml_pipeline.py
```

This will:
1. ✅ Load and analyze the train/val/test dataset
2. ✅ Extract color, texture, shape, and statistical features
3. ✅ Apply multiple feature selection methods (SelectKBest, PCA, RFE)
4. ✅ Train 6 different classifiers (Decision Tree, RF, XGBoost, KNN, SVM, ANN)
5. ✅ Evaluate with accuracy, precision, recall, F1-score, and confusion matrices
6. ✅ Generate comprehensive visualizations and comparison plots
7. ✅ Save trained models for use in the UI

**Output**: All results saved to `outputs/ml_pipeline/`

### Option 3: YOLO Deep Learning Approach

Run the YOLO-based detection pipeline:

```bash
python ppe.py
```

This will:
- Download dataset (if not already present)
- Analyze dataset statistics
- Generate label visualizations
- Train YOLOv8 model
- Validate and predict on test images
- Save trained model to `models/ppe_yolov8.pt`

### Option 4: Run Inference Only

For quick inference with trained YOLO model:

```bash
python infer.py
```

Or for ONNX format:

```bash
python inferonnx.py
```

## Results

After running `ml_pipeline.py`, you'll find:

### Outputs in `outputs/ml_pipeline/`:

1. **Confusion Matrices**: `confusion_matrix_<method>_<classifier>.png`
   - Visual representation of true vs predicted labels
   - One for each classifier-feature method combination
   
2. **Metrics Comparison**: `metrics_comparison.png`
   - Bar plots comparing Accuracy, Precision, Recall, F1-Score
   - Across all classifiers and feature selection methods
   
3. **Detailed Metrics**: `detailed_metrics.csv`
   - Complete numerical results
   - Easy to import into Excel/Google Sheets for further analysis
   
4. **Cached Features**: `features_train.pkl`, `features_test.pkl`
   - Extracted features saved for quick re-runs
   - Delete these to force re-extraction

### Sample Results Format:

```
SUMMARY: Best Performing Models
================================================
Best Accuracy:
  Classifier: Random Forest
  Feature Method: pca
  Accuracy: 0.8523

Best F1-Score:
  Classifier: XGBoost
  Feature Method: kbest
  F1-Score: 0.8467
```

## Key Features

### Comprehensive Feature Extraction
- **160 color histogram features** (BGR + HSV)
- **9 texture features** (Sobel, Laplacian, gray statistics)
- **8 shape features** (contours, edges)
- **15 statistical features** (per-channel statistics)
- **Total: ~192 features per image**

### Robust Feature Selection
- Reduces dimensionality while maintaining performance
- Compares 3 different selection strategies
- Prevents overfitting on high-dimensional data

### Multiple Classifier Comparison
- **Tree-based**: Decision Tree, Random Forest, XGBoost
- **Instance-based**: KNN
- **Kernel-based**: SVM
- **Neural Network**: Multi-layer Perceptron
- Find the best model for your specific use case

### Comprehensive Evaluation
- Multiple metrics beyond just accuracy
- Confusion matrices for error analysis
- Visual comparisons across all combinations
- Exportable results for reports/presentations

## Verification Checklist

- [x] **I. Dataset Split**: ✅ Train (1132) / Val (143) / Test (141)
- [x] **II. Feature Extraction**: ✅ Color, Texture, Shape, Statistical features
- [x] **III. Feature Selection**: ✅ SelectKBest, PCA, RFE
- [x] **IV. Multiple Classifiers**: ✅ Decision Tree, RF, XGBoost, KNN, SVM, ANN
- [x] **V. Evaluation Metrics**: ✅ Accuracy, Precision, Recall, F1, Confusion Matrix

## Troubleshooting

### Dataset Download Issues

**Problem**: Dataset download fails or times out

**Solution**:
```bash
# Try manual download
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
unzip construction-ppe.zip -d datasets/
```

Or use curl:
```bash
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip -o construction-ppe.zip
unzip construction-ppe.zip -d datasets/
```

### UI Not Loading / "YOLO model not found"

**Problem**: UI shows "model not found" error

**Solution**: Train the models first:
```bash
# Train YOLO model
python ppe.py

# Train traditional ML models
python ml_pipeline.py
```

### UI Port Already in Use

**Problem**: Port 8501 is already in use

**Solution**: Specify a different port:
```bash
streamlit run app.py --server.port 8502
```

### Out of Memory

If you encounter memory issues:
- Reduce image size in `extract_all_features()` (default: 128x128)
- Process fewer images for testing
- Use feature selection methods that reduce dimensionality more

### Slow Training

- SVM and ANN can be slow on large datasets
- Start with Random Forest or XGBoost for faster results
- Use cached features (`.pkl` files) for subsequent runs

### Missing Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Streamlit Not Found

If `streamlit run app.py` fails:
```bash
pip install streamlit
# or
pip install --upgrade streamlit
```

## Using the Web UI

After launching the UI with `streamlit run app.py`:

### 1. Upload an Image
- Click "Browse files" or drag and drop an image
- Supported formats: JPG, JPEG, PNG
- Best results with clear images showing workers and PPE

### 2. Choose Detection Mode

**YOLO Object Detection:**
- Detects and localizes multiple PPE items in the image
- Shows bounding boxes around detected objects
- Provides confidence scores for each detection
- Best for: Images with multiple workers/objects
- Detects: Helmets, gloves, vests, boots, goggles (safe and unsafe)

**Traditional ML Classification:**
- Classifies entire image for overall safety assessment
- Choose from 6+ different classifiers:
  - **Random Forest**: Ensemble method, usually best accuracy
  - **XGBoost**: Gradient boosting, fast and accurate
  - **SVM**: Support Vector Machine, good for complex boundaries
  - **Decision Tree**: Simple, interpretable
  - **KNN**: Instance-based learning
  - **ANN (MLP)**: Neural network approach
- Best for: Overall safety compliance checking

### 3. View Results

**YOLO Results:**
- Annotated image with color-coded bounding boxes:
  - 🟢 Green: Safe PPE (helmet, gloves, vest, boots, goggles)
  - 🔴 Red: Missing/unsafe PPE (no_helmet, no_gloves, etc.)
  - 🟡 Yellow: Person detection
- List of all detected items with confidence scores
- Safety summary (compliant/non-compliant)

**ML Classification Results:**
- Safety assessment: "✅ Safe PPE" or "⚠️ Unsafe PPE"
- Confidence score (0-100%)
- Predicted class label
- Feature importance (for tree-based models)

### 4. Compare Models
- Upload the same image multiple times
- Try different classifiers
- Compare accuracy and confidence
- Choose the best model for your use case

## Docker Deployment

The project includes Docker support for easy deployment:

```bash
# Build the Docker image
docker build -t ppe-detection .

# Run the container
docker run -p 8501:8501 ppe-detection
```

Access the UI at `http://localhost:8501`

**Note**: The Docker configuration is optimized for Hugging Face Spaces deployment.

## Extensions and Improvements

Potential enhancements:
1. **Deep Features**: Extract features from pre-trained CNNs (ResNet, VGG)
2. **Hyperparameter Tuning**: GridSearchCV for optimal parameters
3. **Ensemble Methods**: Combine multiple classifiers with voting
4. **Class Imbalance**: Handle with SMOTE or class weights
5. **Cross-Validation**: K-fold CV for more robust evaluation

## License

This project uses the Ultralytics YOLO model which is licensed under AGPL-3.0.
Dataset: Construction-PPE by Ultralytics

## References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**Author**: Mohamd Tobgi  
**Course**: Introduction to Machine Learning - Fall 2026  
**Project**: PPE Detection with Traditional ML Pipeline
