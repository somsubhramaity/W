import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

# ---- Config ----
MODEL_PATH = "models/ppe_yolov8.onnx"   # change if needed
IMAGE_PATH = "/Users/mohamdtobgi/Fall26/intro to ml/PPL detection/outputs/label_vis_train/image440.jpg"              # change to any image
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_THRES = 0.50

NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots",
}

# ---- Helpers ----
def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """Resize with unchanged aspect ratio using padding (like YOLO)."""
    shape = im.shape[:2]  # (h,w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w,h)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, (left, top)

def xywh_to_xyxy(xywh):
    # xywh: [x_center, y_center, w, h]
    x, y, w, h = xywh
    return np.array([x - w/2, y - h/2, x + w/2, y + h/2], dtype=np.float32)

def iou(box, boxes):
    # box: (4,), boxes: (N,4) all in xyxy
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2]-box[0]) * (box[3]-box[1])
    area2 = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union

def nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        overlaps = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][overlaps < iou_thres]
    return keep

# ---- Main ----
def main():
    img0 = cv2.imread(IMAGE_PATH)
    if img0 is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    h0, w0 = img0.shape[:2]
    img, r, (padx, pady) = letterbox(img0, IMG_SIZE)

    # BGR->RGB, normalize, CHW, batch
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_in = np.transpose(img_in, (2, 0, 1))
    img_in = np.expand_dims(img_in, 0)

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: img_in})[0]  # (1, 15, 8400) typically

    # Make it (8400, 15)
    pred = np.squeeze(pred, axis=0).transpose(1, 0)

    # Split
    boxes_xywh = pred[:, 0:4]
    cls_scores = pred[:, 4:]  # (8400, 11)

    cls_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]

    # Filter by confidence
    mask = confs >= CONF_THRES
    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    if boxes_xywh.shape[0] == 0:
        print("No detections above confidence threshold.")
        return

    # Convert boxes to xyxy (in letterboxed image coords)
    boxes_xyxy = np.array([xywh_to_xyxy(b) for b in boxes_xywh])

    # NMS per class (better)
    final = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep = nms(boxes_xyxy[idx], confs[idx], IOU_THRES)
        for k in keep:
            final.append((boxes_xyxy[idx][k], float(confs[idx][k]), int(c)))

    # Sort by confidence desc
    final.sort(key=lambda x: x[1], reverse=True)

    # Map boxes back to original image size (undo padding + scaling)
    out_img = img0.copy()
    for box, conf, c in final:
        x1, y1, x2, y2 = box
        # remove padding
        x1 -= padx; x2 -= padx
        y1 -= pady; y2 -= pady
        # scale back
        x1 /= r; x2 /= r
        y1 /= r; y2 /= r

        # clip
        x1 = int(max(0, min(w0-1, x1)))
        x2 = int(max(0, min(w0-1, x2)))
        y1 = int(max(0, min(h0-1, y1)))
        y2 = int(max(0, min(h0-1, y2)))

        label = f"{NAMES.get(c, c)} {conf:.2f}"
        cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out_img, label, (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = Path("outputs")
    out_path.mkdir(exist_ok=True)
    save_path = out_path / "onnx_result.jpg"
    cv2.imwrite(str(save_path), out_img)

    print(f"Saved: {save_path}")
    print("Top detections:")
    for box, conf, c in final[:10]:
        print(NAMES.get(c, c), conf, box.tolist())

if __name__ == "__main__":
    main()
