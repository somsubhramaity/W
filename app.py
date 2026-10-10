"""
PPE Detection App - YOLO Detection + ML Classification
Combines YOLO object detection with traditional ML classifiers
"""

import gc
import json
import tempfile
import time
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import cv2
import numpy as np
import pickle
from pathlib import Path
from PIL import Image

from alerts import AlertManager
from pipeline import BASE_DIR, FactorySafetyPipeline, load_config
from video_summary import VideoSummaryTracker, save_summary, send_summary_telegram

# Configuration
OUTPUT_DIR = Path("outputs/ml_pipeline")
YOLO_MODEL_PATH = Path("models/ppe_yolov8.pt")

# Class colors (BGR format for OpenCV)
CLASS_COLORS = {
    'helmet': (0, 255, 0), 'gloves': (0, 255, 0), 'vest': (0, 255, 0),
    'boots': (0, 255, 0), 'goggles': (0, 255, 0),
    'no_helmet': (0, 0, 255), 'no_goggle': (0, 0, 255),
    'no_gloves': (0, 0, 255), 'no_boots': (0, 0, 255),
    'Person': (255, 255, 0), 'none': (128, 128, 128)
}


def _cleanup_uploaded_video(capture, video_path):
    if capture is not None:
        capture.release()
    capture = None
    gc.collect()

    if not video_path:
        return
    path = Path(video_path)
    for attempt in range(5):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == 4:
                print(f"Temporary video is still in use; leaving it for later cleanup: {path}")
                return
            time.sleep(0.1)

@st.cache_resource(ttl=60)  # Refresh every 60 seconds during debugging
def load_models():
    """Load all models and feature selectors."""
    models = {}
    
    # Check if output directory exists
    if not OUTPUT_DIR.exists():
        st.error(f"Output directory not found: {OUTPUT_DIR}")
        st.info("Please ensure the models directory is properly deployed.")
        return {}, {}, None, None, None
    
    # Feature methods available
    feature_methods = ['original', 'kbest', 'pca', 'rfe']
    
    # Classifier names
    classifier_names = ['Decision Tree', 'Random Forest', 'XGBoost', 'KNN', 'SVM', 'ANN (MLP)', 'Extra Trees', 'Gradient Boost']
    
    # Load all model combinations
    for clf_name in classifier_names:
        safe_clf = clf_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        for feat_method in feature_methods:
            model_key = f"{clf_name} ({feat_method})"
            model_path = OUTPUT_DIR / f"{safe_clf}_model_{feat_method}.pkl"
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        models[model_key] = pickle.load(f)
                        print(f"Loaded model: {model_key}")
                except Exception as e:
                    print(f"Error loading {model_key}: {e}")
            else:
                print(f"Model not found: {model_path}")
    
    # Add YOLO as an option
    models['YOLO'] = 'YOLO'
    
    # Load feature selectors
    selectors = {'original': None}  # Original has no selector
    for feat_method in ['kbest', 'pca', 'rfe']:
        selector_path = OUTPUT_DIR / f"selector_{feat_method}.pkl"
        if selector_path.exists():
            try:
                with open(selector_path, 'rb') as f:
                    selectors[feat_method] = pickle.load(f)
                    print(f"Loaded selector: {feat_method}")
            except Exception as e:
                print(f"Error loading selector {feat_method}: {e}")
                selectors[feat_method] = None
        else:
            print(f"Warning: Selector not found: {selector_path}")
            selectors[feat_method] = None
    
    # Load scaler and label encoder
    try:
        scaler_path = OUTPUT_DIR / 'scaler.pkl'
        encoder_path = OUTPUT_DIR / 'label_encoder.pkl'
        
        if not scaler_path.exists() or not encoder_path.exists():
            st.error("Required files (scaler.pkl or label_encoder.pkl) not found")
            return models, selectors, None, None, None
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler or encoder: {e}")
        return models, selectors, None, None, None
    
    yolo_model = YOLO(str(YOLO_MODEL_PATH)) if YOLO_MODEL_PATH.exists() else None
    
    return models, selectors, scaler, label_encoder, yolo_model

def extract_hog_features(img, target_size=(64, 64)):
    """Extract HOG (Histogram of Oriented Gradients) features."""
    img = cv2.resize(img, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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
    
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        features.extend(hist.flatten())
    
    for i, (bins, range_max) in enumerate([(32, 180), (32, 256), (32, 256)]):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, range_max])
        features.extend(hist.flatten())
    
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [32], [0, 256])
        features.extend(hist.flatten())
    
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
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    
    features.extend([
        np.mean(magnitude), np.std(magnitude), np.max(magnitude),
        np.mean(direction), np.std(direction)
    ])
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(np.var(laplacian))
    
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for sigma in [1, 3]:
            kernel = cv2.getGaborKernel((21, 21), sigma, theta, 10, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            features.extend([np.mean(filtered), np.std(filtered)])
    
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
    """Extract ALL enhanced features from ROI."""
    roi = cv2.resize(roi, target_size)
    
    hog = extract_hog_features(roi, target_size)
    lbp = extract_lbp_features(roi, target_size)
    color = extract_color_features(roi, target_size)
    texture = extract_texture_features(roi, target_size)
    shape = extract_shape_features(roi, target_size)
    
    features = np.concatenate([hog, lbp, color, texture, shape])
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
    
    return features

def detect_and_classify(image, classifier, feature_selector, scaler, label_encoder, yolo_model, conf_threshold=0.25, use_yolo_class=False):
    """Detect with YOLO and classify with ML model or YOLO."""
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_display = img.copy()
        detections = []
        
        # YOLO class names mapping
        yolo_class_names = {
            0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
            5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
            9: "no_gloves", 10: "no_boots"
        }
        
        if yolo_model is None:
            st.error("YOLO model not loaded")
            return img_display, detections
    except Exception as e:
        st.error(f"Error initializing detection: {e}")
        import traceback
        st.code(traceback.format_exc())
        return np.array(image), []
    
    try:
        if yolo_model:
            results = yolo_model(img, conf=conf_threshold, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    try:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        roi = img[y1:y2, x1:x2]
                        
                        if roi.size > 0 and roi.shape[0] > 20 and roi.shape[1] > 20:
                            # Determine class and confidence
                            if use_yolo_class:
                                # Use YOLO's prediction
                                yolo_cls = int(box.cls[0])
                                pred_class = yolo_class_names.get(yolo_cls, f"class_{yolo_cls}")
                                pred_conf = float(box.conf[0])
                            else:
                                # Use ML classifier with feature selection
                                features = extract_all_features(roi)
                                features_scaled = scaler.transform(features.reshape(1, -1))
                                
                                # Apply feature selection if provided
                                if feature_selector is not None:
                                    features_scaled = feature_selector.transform(features_scaled)
                                
                                # Debug: print feature dimensions
                                print(f"Features shape after selection: {features_scaled.shape}")
                                
                                pred_encoded = classifier.predict(features_scaled)[0]
                                pred_class = label_encoder.inverse_transform([pred_encoded])[0]
                                
                                if hasattr(classifier, 'predict_proba'):
                                    proba = classifier.predict_proba(features_scaled)[0]
                                    pred_conf = float(proba[pred_encoded])
                                else:
                                    pred_conf = float(box.conf[0])
                            
                            detections.append({
                                'box': (x1, y1, x2, y2),
                                'class': pred_class,
                                'confidence': pred_conf
                            })
                            
                            # Draw
                            color = CLASS_COLORS.get(pred_class, (255, 255, 255))
                            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 3)
                            
                            label = f"{pred_class}: {pred_conf:.2f}"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(img_display, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                            cv2.putText(img_display, label, (x1, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    except Exception as e:
                        st.warning(f"Error processing detection box: {e}")
                        continue
        
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        return img_display, detections
    except Exception as e:
        st.error(f"Error during detection: {e}")
        import traceback
        st.code(traceback.format_exc())
        return np.array(image), []

def main():
    st.set_page_config(page_title="PPE Detection", page_icon="🦺", layout="wide")
    
    st.title("🦺 PPE Detection: YOLO + ML Classifiers")
    st.markdown("""
    **How it works:**
    1. YOLO detects objects and bounding boxes
    2. Your chosen ML classifier identifies each object
    3. Results visualized with colored boxes
    """)
    
    # Load models
    with st.spinner("Loading models..."):
        try:
            models, selectors, scaler, label_encoder, yolo_model = load_models()
            if not models or not yolo_model:
                st.error("Models not found! Run: `python3 ml_pipeline.py`")
                return
            st.success(f"✅ Loaded {len(models)} model combinations")
        except Exception as e:
            st.error(f"Error: {e}")
            return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Feature selection method dropdown
    feature_method = st.sidebar.selectbox(
        "Feature Selection Method",
        options=['original', 'kbest', 'pca', 'rfe'],
        format_func=lambda x: {
            'original': 'Original (All Features - 2361)',
            'kbest': 'SelectKBest (Top 200)',
            'pca': 'PCA (150 Components)',
            'rfe': 'RFE (150 Features)'
        }[x]
    )
    
    # Classifier dropdown
    classifier_names = ['Decision Tree', 'Random Forest', 'XGBoost', 'KNN', 'SVM', 'ANN (MLP)', 'Extra Trees', 'Gradient Boost', 'YOLO']
    selected_classifier = st.sidebar.selectbox("Classifier", classifier_names)
    
    # Construct full model key
    if selected_classifier == 'YOLO':
        selected_model = 'YOLO'
        use_yolo = True
    else:
        selected_model = f"{selected_classifier} ({feature_method})"
        use_yolo = False
    
    conf_threshold = 0.25  # Fixed confidence threshold
    
    # Upload
    uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        try:
            st.write("Debug: File uploaded successfully")
            image = Image.open(uploaded_file)
            st.write(f"Debug: Image opened, size: {image.size}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original")
                st.image(image, use_container_width=True)
            
            with st.spinner(f"Processing with {selected_classifier} ({feature_method if not use_yolo else 'N/A'})..."):
                # Get feature selector based on selected method
                feature_selector = selectors.get(feature_method, None) if not use_yolo else None
                
                # Debug info
                if not use_yolo:
                    st.sidebar.write(f"🔍 Using: {selected_classifier} with {feature_method} features")
                    st.sidebar.write(f"🔍 Selector: {type(feature_selector).__name__ if feature_selector else 'None (original features)'}")
                
                # Check if model exists
                if selected_model not in models:
                    st.error(f"Model '{selected_model}' not found! Run ml_pipeline.py first.")
                    st.stop()
                
                st.write(f"Debug: About to call detect_and_classify")
                result_img, detections = detect_and_classify(
                    image, 
                    models[selected_model] if not use_yolo else None,
                    feature_selector,
                    scaler, 
                    label_encoder, 
                    yolo_model, 
                    conf_threshold,
                    use_yolo_class=use_yolo
                )
                st.write(f"Debug: detect_and_classify completed. Found {len(detections)} detections")
                
                with col2:
                    st.subheader(f"🔍 Results ({selected_model})")
                    st.image(result_img, use_container_width=True)
            
            # Stats
            st.markdown("---")
            if detections:
                cols = st.columns(4)
                
                with cols[0]:
                    st.metric("Total Detections", len(detections))
                
                class_counts = {}
                for d in detections:
                    class_counts[d['class']] = class_counts.get(d['class'], 0) + 1
                
                safe = sum(class_counts.get(c, 0) for c in ['helmet', 'gloves', 'vest', 'boots', 'goggles'])
                unsafe = sum(class_counts.get(c, 0) for c in ['no_helmet', 'no_goggle', 'no_gloves', 'no_boots'])
                
                with cols[1]:
                    st.metric("✅ Safe PPE", safe)
                with cols[2]:
                    st.metric("⚠️ Missing PPE", unsafe, delta=f"-{unsafe}" if unsafe > 0 else None, delta_color="inverse")
                with cols[3]:
                    safety_score = (safe / (safe + unsafe) * 100) if (safe + unsafe) > 0 else 0
                    st.metric("Safety Score", f"{safety_score:.0f}%")
                
                # Table
                st.subheader("📊 Detected Objects")
                import pandas as pd
                df = pd.DataFrame(detections)
                df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
                df = df[['class', 'confidence']]
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No objects detected")
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("👆 Upload an image to start!")

def factory_safety_main():
    st.set_page_config(page_title="Factory Safety Monitor", page_icon="FS", layout="wide")
    st.title("Factory Safety Monitor")
    st.caption("Live PPE compliance, smoke, and fire monitoring")

    try:
        config = load_config()
    except Exception as error:
        st.error(f"Could not load config.yaml: {error}")
        return

    if "running" not in st.session_state:
        st.session_state.running = False
        st.session_state.alert_rows = []
        st.session_state.people_seen = 0
        st.session_state.people_compliant = 0
        st.session_state.violation_count = 0
        st.session_state.hazard_count = 0
        summary_path = BASE_DIR / "outputs" / "summary.json"
        try:
            with summary_path.open("r", encoding="utf-8") as summary_file:
                st.session_state.video_summary = json.load(summary_file)
            st.session_state.summary_path = str(summary_path)
        except (OSError, json.JSONDecodeError):
            st.session_state.video_summary = None

    with st.sidebar:
        st.header("Camera")
        source_type = st.selectbox("Source", ["Upload video", "Webcam", "RTSP stream"])
        camera_id = st.text_input("Camera ID", "camera-1").strip() or "camera-1"
        default_zone = config.get("camera_zones", {}).get(camera_id, "Factory Floor")
        zone = st.text_input("Zone / area", default_zone)
        required_gear = st.multiselect(
            "Required gear",
            ["helmet", "vest", "gloves", "boots"],
            default=config.get("required_gear", ["helmet", "vest"]),
        )
        uploaded_video = None
        rtsp_url = ""
        if source_type == "Upload video":
            uploaded_video = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
        elif source_type == "RTSP stream":
            rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://user:password@host:554/stream")

        start_clicked = st.button("Start monitoring", disabled=st.session_state.running, type="primary")
        stop_clicked = st.button("Stop monitoring", disabled=not st.session_state.running)

    if stop_clicked:
        capture = st.session_state.pop("capture", None)
        temp_video = st.session_state.pop("temp_video", None)
        _cleanup_uploaded_video(capture, temp_video)
        st.session_state.running = False
        st.session_state.pop("pipeline", None)
        st.session_state.pop("alert_manager", None)

    if start_clicked:
        source = None
        temp_video = None
        if source_type == "Webcam":
            source = 0
        elif source_type == "RTSP stream":
            source = rtsp_url.strip()
            if not source:
                st.sidebar.error("Enter an RTSP URL first.")
        elif uploaded_video is not None:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            temp_dir = BASE_DIR / "outputs"
            temp_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix="uploaded_source_",
                suffix=suffix,
                dir=str(temp_dir),
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_video.getvalue())
                temp_path = Path(temp_file.name)
            source = str(temp_path)
            temp_video = str(temp_path)
        else:
            st.sidebar.error("Choose a video file before starting.")

        if source is not None:
            capture = None
            try:
                pipeline = FactorySafetyPipeline()
                pipeline.required_gear = set(required_gear)
                capture = cv2.VideoCapture(source)
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"Could not open source: {source_type}")
                st.session_state.capture = capture
                st.session_state.pipeline = pipeline
                st.session_state.alert_manager = AlertManager(config=config)
                st.session_state.camera_id = camera_id
                st.session_state.zone = zone
                st.session_state.source_type = source_type
                st.session_state.temp_video = temp_video
                st.session_state.video_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                st.session_state.video_frame_count = 0
                st.session_state.video_summary = None
                st.session_state.summary_telegram_result = None
                st.session_state.summary_tracker = (
                    VideoSummaryTracker(
                        video_path=uploaded_video.name,
                        camera_id=camera_id,
                        zone=zone,
                        config=config,
                    )
                    if source_type == "Upload video"
                    else None
                )
                st.session_state.alert_rows = []
                st.session_state.people_seen = 0
                st.session_state.people_compliant = 0
                st.session_state.violation_count = 0
                st.session_state.hazard_count = 0
                st.session_state.running = True
            except Exception as error:
                st.session_state.pop("capture", None)
                _cleanup_uploaded_video(capture, temp_video)
                st.error(f"Could not start monitoring: {error}")

    @st.fragment(run_every=0.2)
    def render_monitor():
        capture = st.session_state.get("capture")
        if capture is None or not st.session_state.get("running", False):
            return
        ok, frame = capture.read()
        if not ok:
            if st.session_state.source_type == "Upload video":
                temp_video = st.session_state.pop("temp_video", None)
                st.session_state.pop("capture", None)
                _cleanup_uploaded_video(capture, temp_video)
                capture = None
                tracker = st.session_state.get("summary_tracker")
                if tracker is not None and st.session_state.video_summary is None:
                    summary = tracker.build(st.session_state.video_fps)
                    summary_path = save_summary(summary)
                    st.session_state.video_summary = summary
                    st.session_state.summary_path = str(summary_path)
                    st.session_state.summary_telegram_result = send_summary_telegram(summary)
                st.session_state.running = False
                st.rerun()
            else:
                st.warning("No frame received. The stream will be retried.")
            return

        annotated, detections, frame_info = st.session_state.pipeline.process_frame(
            frame,
            camera_id=st.session_state.camera_id,
            zone=st.session_state.zone,
        )
        new_alerts = st.session_state.alert_manager.process_frame(
            detections,
            annotated,
            inference_performed=frame_info["inference_performed"],
            camera_id=st.session_state.camera_id,
            hazard_inference_performed=frame_info["hazard_inference_performed"],
        )
        st.session_state.alert_rows = new_alerts + st.session_state.alert_rows
        st.session_state.hazard_count += sum(
            alert["event_type"] in ("FIRE", "SMOKE") for alert in new_alerts
        )
        if frame_info["inference_performed"]:
            for person in frame_info["persons"]:
                st.session_state.people_seen += 1
                if person["status"] == "compliant":
                    st.session_state.people_compliant += 1
                else:
                    st.session_state.violation_count += 1
        summary_tracker = st.session_state.get("summary_tracker")
        if summary_tracker is not None:
            st.session_state.video_frame_count += 1
            summary_tracker.update(
                frame_index=st.session_state.video_frame_count,
                video_time_seconds=(st.session_state.video_frame_count - 1) / st.session_state.video_fps,
                persons=frame_info["persons"],
                detections=detections,
                frame=frame,
                hazard_inference_performed=frame_info["hazard_inference_performed"],
            )

        active_hazard = any(item["event_type"] in ("FIRE", "SMOKE") for item in detections)
        if active_hazard:
            st.markdown(
                "<div style='padding:10px 14px;background:#b42318;color:white;font-weight:700'>"
                "FIRE / SMOKE DETECTED</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='padding:10px 14px;background:#176b48;color:white;font-weight:700'>"
                "No active fire or smoke alert</div>",
                unsafe_allow_html=True,
            )

        left, right = st.columns([3, 2])
        with left:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        with right:
            total = st.session_state.people_seen
            compliance = 100 * st.session_state.people_compliant / total if total else 0
            metric_columns = st.columns(2)
            metric_columns[0].metric("Compliance", f"{compliance:.1f}%")
            metric_columns[1].metric("PPE violations", st.session_state.violation_count)
            metric_columns = st.columns(2)
            metric_columns[0].metric("Fire / smoke alerts", st.session_state.hazard_count)
            metric_columns[1].metric("Live FPS", f"{frame_info['fps']:.1f}")
            st.caption(
                f"{st.session_state.camera_id} · {st.session_state.zone} · "
                f"{frame_info['counts']['compliant']} compliant / {frame_info['counts']['violations']} violations in current frame"
            )

        st.subheader("Live alerts")
        if st.session_state.alert_rows:
            headers = st.columns([2, 2, 2, 1, 2])
            for column, label in zip(headers, ["Zone", "Type", "Time", "Severity", "Snapshot"]):
                column.caption(label)
            for alert in st.session_state.alert_rows[:8]:
                columns = st.columns([2, 2, 2, 1, 2])
                columns[0].write(alert["zone"])
                columns[1].write(alert["event_type"])
                columns[2].write(alert["timestamp"])
                columns[3].write(alert["severity"])
                snapshot_path = BASE_DIR / alert["snapshot"]
                if snapshot_path.is_file():
                    columns[4].image(str(snapshot_path), width=96)
        else:
            st.caption("No alerts recorded for this session.")

    if st.session_state.running:
        render_monitor()
    else:
        st.info("Choose a source, assign its zone, then start monitoring.")
        summary = st.session_state.get("video_summary")
        if summary:
            st.subheader("Completed video summary")
            st.caption(
                f"{summary['video_name']} · {summary['camera_id']} · {summary['zone']} · "
                f"{summary['duration']}"
            )
            metric_columns = st.columns(4)
            metric_columns[0].metric("Unique people", summary["total_unique_people"])
            metric_columns[1].metric("WITHOUT helmet", summary["without_helmet"])
            metric_columns[2].metric("With helmet", summary["with_helmet"])
            metric_columns[3].metric("Without vest", summary["without_vest"])
            hazards = summary["fire_smoke_events"]
            st.write(
                f"Fire/smoke detected: {'Yes' if hazards['detected'] else 'No'} "
                f"({hazards['count']} total; fire {hazards['fire']}, smoke {hazards['smoke']})"
            )
            top_violators = summary.get("top_violators", [])
            if top_violators:
                st.caption("Top helmet violators")
                columns = st.columns(min(3, len(top_violators)))
                for column, violator in zip(columns, top_violators):
                    snapshot = violator.get("best_snapshot")
                    snapshot_path = BASE_DIR / snapshot if snapshot else None
                    if snapshot_path and snapshot_path.is_file():
                        column.image(str(snapshot_path), use_container_width=True)
                    else:
                        column.caption("Snapshot unavailable")
                    column.caption(
                        f"Track {violator['track_id']} · "
                        f"helmet missing {violator['helmet_missing_ratio']:.0%} · "
                        f"{violator['frames_seen']} frames"
                    )
            else:
                st.caption("No tracks met the helmet-violation threshold.")
            resend_columns = st.columns([1, 4])
            if resend_columns[0].button("Send summary to Telegram"):
                st.session_state.summary_telegram_result = send_summary_telegram(summary)
            result = st.session_state.get("summary_telegram_result") or {}
            if result.get("sent"):
                resend_columns[1].success("Summary sent to Telegram.")
            elif result.get("credentials_missing"):
                resend_columns[1].warning(
                    "Telegram is not configured. Set TELEGRAM_BOT_TOKEN and either "
                    "TELEGRAM_CHAT_ID or TELEGRAM_CHAT_IDS in .env, then resend."
                )
            elif result.get("error"):
                resend_columns[1].error(result["error"])
            st.caption(f"Saved to {st.session_state.get('summary_path', 'outputs/summary.json')}")


if __name__ == "__main__":
    factory_safety_main()
