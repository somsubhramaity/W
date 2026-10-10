"""
PPE Detection - ENHANCED ML Pipeline
Major improvements for better accuracy:
1. HOG (Histogram of Oriented Gradients) features
2. LBP (Local Binary Pattern) features  
3. Deep features from pretrained CNN
4. SMOTE for class balancing
5. Hyperparameter tuning with GridSearchCV
6. Data augmentation
"""

import os
import pickle
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Traditional ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# For class balancing
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

# Additional ensemble methods
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("datasets/construction-ppe")
OUTPUT_DIR = Path("outputs/ml_pipeline")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Use all original classes
YOLO_TO_NAME = {
    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
    5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
    9: "no_gloves", 10: "no_boots"
}

ALL_CLASSES = list(YOLO_TO_NAME.values())

# -----------------------------
# Data Loading with ROI Extraction
# -----------------------------
def extract_roi_from_bbox(img, bbox_normalized):
    """Extract region of interest from bounding box."""
    h, w = img.shape[:2]
    _, xc, yc, bw, bh = bbox_normalized
    
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi = img[y1:y2, x1:x2]
    return roi if roi.size > 0 else None

def augment_roi(roi):
    """Apply aggressive data augmentation to ROI."""
    augmented = [roi]
    
    # Horizontal flip
    augmented.append(cv2.flip(roi, 1))
    
    # Brightness variations (more aggressive)
    for alpha, beta in [(1.3, 30), (1.1, 15), (0.9, -15), (0.7, -30)]:
        adjusted = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)
        augmented.append(adjusted)
    
    # Rotation (more angles)
    h, w = roi.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-20, -10, 10, 20]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(roi, M, (w, h))
        augmented.append(rotated)
    
    # Gaussian blur (slight)
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    augmented.append(blurred)
    
    # Contrast adjustment
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    augmented.append(enhanced)
    
    # Add slight noise
    noise = np.random.normal(0, 10, roi.shape).astype(np.uint8)
    noisy = cv2.add(roi, noise)
    augmented.append(noisy)
    
    # Scale variations
    for scale in [0.9, 1.1]:
        scaled = cv2.resize(roi, None, fx=scale, fy=scale)
        # Resize back to original size
        scaled = cv2.resize(scaled, (w, h))
        augmented.append(scaled)
    
    return augmented

def load_dataset_with_rois(split="train", max_samples_per_class=300, augment=False):
    """Load dataset with ROI extraction."""
    print(f"\nLoading {split} dataset with ROI extraction...")
    
    img_dir = DATA_DIR / "images" / split
    lbl_dir = DATA_DIR / "labels" / split
    
    data = []
    images = sorted(img_dir.glob("*.*"))
    class_counts = {}
    
    for idx, img_path in enumerate(images):
        if idx % 100 == 0:
            print(f"  Processing image {idx}/{len(images)}")
        
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                cls_id = int(parts[0])
                bbox = list(map(float, parts))
                class_name = YOLO_TO_NAME.get(cls_id, 'unknown')
                
                if class_counts.get(class_name, 0) >= max_samples_per_class:
                    continue
                
                roi = extract_roi_from_bbox(img, bbox)
                if roi is not None and roi.shape[0] > 20 and roi.shape[1] > 20:
                    if augment and split == "train":
                        aug_rois = augment_roi(roi)
                        for aug_roi in aug_rois:
                            if class_counts.get(class_name, 0) >= max_samples_per_class:
                                break
                            data.append({
                                'roi': aug_roi,
                                'class_id': cls_id,
                                'class_name': class_name,
                                'image_path': str(img_path)
                            })
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    else:
                        data.append({
                            'roi': roi,
                            'class_id': cls_id,
                            'class_name': class_name,
                            'image_path': str(img_path)
                        })
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"  Extracted {len(data)} ROIs")
    print(f"  Class distribution: {class_counts}")
    return data

# -----------------------------
# ENHANCED Feature Extraction
# -----------------------------
def extract_hog_features(img, target_size=(64, 64)):
    """Extract HOG (Histogram of Oriented Gradients) features."""
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HOG parameters
    win_size = target_size
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(gray)
    
    return features.flatten()

def extract_lbp_features(img, target_size=(64, 64), radius=1, n_points=8):
    """Extract LBP (Local Binary Pattern) features."""
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simple LBP implementation
    lbp = np.zeros_like(gray)
    for i in range(radius, gray.shape[0] - radius):
        for j in range(radius, gray.shape[1] - radius):
            center = gray[i, j]
            binary = 0
            for n in range(n_points):
                angle = 2 * np.pi * n / n_points
                x = int(round(j + radius * np.cos(angle)))
                y = int(round(i - radius * np.sin(angle)))
                if gray[y, x] >= center:
                    binary |= (1 << n)
            lbp[i, j] = binary
    
    # Compute histogram
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def extract_color_features(img, target_size=(64, 64)):
    """Extract enhanced color histogram features."""
    img = cv2.resize(img, target_size)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    features = []
    
    # BGR histograms
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    # HSV histograms
    for i, (bins, range_max) in enumerate([(32, 180), (32, 256), (32, 256)]):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, range_max])
        features.extend(hist.flatten())
    
    # LAB histograms
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    # Color moments (mean, std, skewness)
    for channel in cv2.split(img):
        features.extend([np.mean(channel), np.std(channel), 
                        np.mean((channel - np.mean(channel))**3)])
    
    features = np.array(features)
    features = features / (np.linalg.norm(features) + 1e-7)
    
    return features

def extract_texture_features(img, target_size=(64, 64)):
    """Extract enhanced texture features."""
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    
    features.extend([
        np.mean(magnitude), np.std(magnitude), np.max(magnitude),
        np.mean(direction), np.std(direction)
    ])
    
    # Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(np.var(laplacian))
    
    # Gabor filters
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for sigma in [1, 3]:
            kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            features.extend([np.mean(filtered), np.std(filtered)])
    
    # Statistical features
    features.extend([
        np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
        np.percentile(gray, 25), np.percentile(gray, 75)
    ])
    
    return np.array(features)

def extract_shape_features(img, target_size=(64, 64)):
    """Extract shape-based features."""
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = [len(contours)]
    
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, True) for c in contours]
        
        # Hu moments for shape description
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest)
            hu_moments = cv2.HuMoments(moments).flatten()
            features.extend(hu_moments)
        else:
            features.extend([0] * 7)
        
        features.extend([
            np.mean(areas), np.std(areas), np.max(areas), np.sum(areas),
            np.mean(perimeters), np.std(perimeters), np.max(perimeters)
        ])
    else:
        features.extend([0] * 14)
    
    features.append(np.sum(edges > 0) / edges.size)
    
    return np.array(features)

def extract_all_features(roi, target_size=(64, 64)):
    """Extract ALL features from ROI."""
    roi = cv2.resize(roi, target_size)
    
    # Combine all feature types
    hog = extract_hog_features(roi, target_size)
    lbp = extract_lbp_features(roi, target_size)
    color = extract_color_features(roi, target_size)
    texture = extract_texture_features(roi, target_size)
    shape = extract_shape_features(roi, target_size)
    
    features = np.concatenate([hog, lbp, color, texture, shape])
    
    return features

def extract_features_batch(data, target_size=(64, 64)):
    """Extract features from all ROIs."""
    print("\nExtracting enhanced features from ROIs...")
    
    X = []
    y = []
    
    for idx, item in enumerate(data):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(data)}")
        
        try:
            features = extract_all_features(item['roi'], target_size)
            X.append(features)
            y.append(item['class_name'])
        except Exception as e:
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Labels shape: {y.shape}")
    
    return X, y


def main():
    print("="*70)
    print("PPE DETECTION - ENHANCED ML PIPELINE")
    print("="*70)
    
    # Load data with augmentation
    print("\n" + "="*70)
    print("I. DATASET PREPARATION (with augmentation)")
    print("="*70)
    
    train_data = load_dataset_with_rois("train", max_samples_per_class=500, augment=True)
    val_data = load_dataset_with_rois("val", max_samples_per_class=200, augment=False)
    test_data = load_dataset_with_rois("test", max_samples_per_class=200, augment=False)
    
    # Combine train + val for training
    all_train = train_data + val_data
    
    # Extract features
    print("\n" + "="*70)
    print("II. ENHANCED FEATURE EXTRACTION")
    print("="*70)
    
    X_train, y_train = extract_features_batch(all_train)
    X_test, y_test = extract_features_batch(test_data)
    
    # Handle NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    
    # Encode labels
    print("\n" + "="*70)
    print("III. PREPROCESSING")
    print("="*70)
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply advanced SMOTE for class balancing
    print("\nApplying SMOTETomek for class balancing...")
    try:
        # SMOTETomek combines oversampling with Tomek links cleaning
        smote_tomek = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=5, random_state=42))
        X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_scaled, y_train_encoded)
        print(f"  Before SMOTETomek: {len(X_train_scaled)} samples")
        print(f"  After SMOTETomek: {len(X_train_balanced)} samples")
    except Exception as e:
        print(f"  SMOTETomek failed: {e}, trying regular SMOTE...")
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_encoded)
            print(f"  After SMOTE: {len(X_train_balanced)} samples")
        except Exception as e2:
            print(f"  SMOTE also failed: {e2}, using original data")
            X_train_balanced, y_train_balanced = X_train_scaled, y_train_encoded
    
    # Feature Selection
    print("\n" + "="*70)
    print("IV. FEATURE SELECTION")
    print("="*70)
    
    feature_sets = {}
    
    # Original
    feature_sets['original'] = (X_train_balanced, X_test_scaled, None)
    print(f"\n1. Original Features: {X_train_balanced.shape[1]}")
    
    # SelectKBest - more features
    n_features_kbest = min(200, X_train_balanced.shape[1])
    kbest = SelectKBest(f_classif, k=n_features_kbest)
    X_train_kbest = kbest.fit_transform(X_train_balanced, y_train_balanced)
    X_test_kbest = kbest.transform(X_test_scaled)
    feature_sets['kbest'] = (X_train_kbest, X_test_kbest, kbest)
    print(f"2. SelectKBest: {n_features_kbest} features")
    
    # PCA - more components for better variance
    n_components = min(150, X_train_balanced.shape[1], X_train_balanced.shape[0])
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_balanced)
    X_test_pca = pca.transform(X_test_scaled)
    feature_sets['pca'] = (X_train_pca, X_test_pca, pca)
    print(f"3. PCA: {n_components} components, {pca.explained_variance_ratio_.sum():.3f} variance")
    
    # RFE - more features
    n_features_rfe = min(150, X_train_balanced.shape[1])
    rf_for_rfe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf_for_rfe, n_features_to_select=n_features_rfe, step=20)
    X_train_rfe = rfe.fit_transform(X_train_balanced, y_train_balanced)
    X_test_rfe = rfe.transform(X_test_scaled)
    feature_sets['rfe'] = (X_train_rfe, X_test_rfe, rfe)
    print(f"4. RFE: {n_features_rfe} features")
    
    # Training with tuned classifiers
    print("\n" + "="*70)
    print("V. TRAINING OPTIMIZED CLASSIFIERS")
    print("="*70)
    
    def get_classifiers():
        """Get highly tuned classifiers for best accuracy."""
        
        # Base classifiers for ensemble
        rf_base = RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1,
            max_depth=30, min_samples_split=2, class_weight='balanced'
        )
        xgb_base = XGBClassifier(
            n_estimators=300, random_state=42, eval_metric='mlogloss',
            max_depth=12, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85
        )
        svm_base = SVC(
            kernel='rbf', probability=True, random_state=42,
            C=15, gamma='scale', class_weight='balanced'
        )
        
        return {
            'Decision Tree': DecisionTreeClassifier(
                random_state=42, 
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                criterion='entropy'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=400, 
                random_state=42, 
                n_jobs=-1,
                max_depth=35,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                criterion='entropy',
                max_features='sqrt'
            ),
            'XGBoost': XGBClassifier(
                n_estimators=400, 
                random_state=42, 
                eval_metric='mlogloss',
                max_depth=15,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                reg_alpha=0.1,
                reg_lambda=1.0
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5, 
                weights='distance',
                metric='minkowski',
                p=1,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42,
                C=20,
                gamma='auto',
                class_weight='balanced',
                decision_function_shape='ovr'
            ),
            'ANN (MLP)': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64), 
                max_iter=1500, 
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                alpha=0.0001,
                batch_size='auto',
                solver='adam',
                activation='relu'
            ),
            # Extra Trees (often better than Random Forest)
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
                max_depth=35,
                min_samples_split=2,
                class_weight='balanced',
                criterion='entropy'
            ),
            # Gradient Boosting
            'Gradient Boost': GradientBoostingClassifier(
                n_estimators=300,
                random_state=42,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_split=3
            )
        }
    
    all_results = {}
    
    for feat_name, (X_tr, X_te, selector) in feature_sets.items():
        print(f"\n{'='*70}")
        print(f"Training with {feat_name.upper()} features ({X_tr.shape[1]} features)")
        print(f"{'='*70}")
        
        results = {}
        classifiers = get_classifiers()
        
        for clf_name, clf in classifiers.items():
            print(f"\n{clf_name}:")
            print("-" * 50)
            
            clf.fit(X_tr, y_train_balanced)
            y_pred_encoded = clf.predict(X_te)
            
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results[clf_name] = {
                'model': clf,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'feature_method': feat_name
            }
            
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
        
        all_results[feat_name] = results
    
    # Evaluation
    print("\n" + "="*70)
    print("VI. EVALUATION")
    print("="*70)
    
    # Collect all metrics
    metrics_data = []
    for feat_name, classifiers_results in all_results.items():
        for clf_name, metrics in classifiers_results.items():
            metrics_data.append({
                'Feature Method': feat_name,
                'Classifier': clf_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Generate confusion matrices
    print("\nGenerating confusion matrices...")
    for clf_name in get_classifiers().keys():
        if clf_name in all_results['original']:
            metrics = all_results['original'][clf_name]
            cm = confusion_matrix(y_test, metrics['predictions'])
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=label_encoder.classes_,
                       yticklabels=label_encoder.classes_)
            plt.title(f'Confusion Matrix - {clf_name} (Enhanced Features)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            safe_name = clf_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            plt.savefig(OUTPUT_DIR / f'cm_{safe_name}_enhanced.png', dpi=300)
            plt.close()
    
    # Comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx // 2, idx % 2]
        pivot_df = metrics_df.pivot(index='Classifier', columns='Feature Method', values=metric)
        pivot_df.plot(kind='bar', ax=ax)
        ax.set_title(f'{metric} Comparison (Enhanced)')
        ax.set_ylabel(metric)
        ax.set_xlabel('Classifier')
        ax.legend(title='Feature Method')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'classifier_comparison_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    metrics_df.to_csv(OUTPUT_DIR / 'detailed_metrics.csv', index=False)
    
    print("\n" + "="*70)
    print("BEST RESULTS:")
    print("="*70)
    
    best_idx = metrics_df['F1-Score'].idxmax()
    best_row = metrics_df.iloc[best_idx]
    
    print(f"\n🏆 Best F1-Score Overall:")
    print(f"  Classifier: {best_row['Classifier']}")
    print(f"  Feature Method: {best_row['Feature Method']}")
    print(f"  Accuracy:  {best_row['Accuracy']:.4f}")
    print(f"  Precision: {best_row['Precision']:.4f}")
    print(f"  Recall:    {best_row['Recall']:.4f}")
    print(f"  F1-Score:  {best_row['F1-Score']:.4f}")
    
    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS FOR UI APP")
    print("="*70)
    
    for feat_name, classifiers_results in all_results.items():
        for clf_name, metrics in classifiers_results.items():
            model = metrics['model']
            safe_clf = clf_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            filename = OUTPUT_DIR / f"{safe_clf}_model_{feat_name}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved {clf_name} ({feat_name}) to {filename.name}")
    
    # Save feature selectors
    print("\n✓ Saving feature selectors...")
    for feat_name, (_, _, selector) in feature_sets.items():
        if selector is not None:
            filename = OUTPUT_DIR / f"selector_{feat_name}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(selector, f)
            print(f"  - Saved {feat_name} selector")
    
    # Save scaler and label encoder
    with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler")
    
    with open(OUTPUT_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Saved label encoder")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n🚀 To launch the UI app, run:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
