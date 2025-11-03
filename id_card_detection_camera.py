
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse

# Lightweight helpers (no TF Object Detection API)
def parse_labelmap(labelmap_path):
    classes = {}
    current_id = None
    current_name = None
    with open(labelmap_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('id:'):
                try:
                    current_id = int(line.split(':')[1].strip())
                except Exception:
                    current_id = None
            elif line.startswith('name:') or line.startswith('display_name:'):
                val = line.split(':', 1)[1].strip().strip('"').strip("'")
                current_name = val
            elif line.startswith('item'):
                current_id = None
                current_name = None
            elif line == '}':
                if current_id is not None and current_name is not None:
                    classes[current_id] = {'id': current_id, 'name': current_name}
                current_id = None
                current_name = None
    if not classes:
        classes = {1: {'id': 1, 'name': 'object'}}
    return classes

def draw_boxes(image, boxes, classes, scores, category_index, min_score, line_thickness=3):
    h, w = image.shape[:2]
    best_idx = int(np.argmax(scores)) if scores is not None and len(scores) > 0 else None
    for i in range(len(boxes)):
        score = float(scores[i]) if scores is not None else 1.0
        if score < min_score:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        cls_id = int(classes[i]) if classes is not None else 1
        cls_name = category_index.get(cls_id, {'name': 'object'})['name']
        color = (0, 255, 0) if (best_idx is not None and i == best_idx) else (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        label = f"{cls_name}: {int(score*100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'

# CLI
parser = argparse.ArgumentParser(description='ID Card Detection (camera, TF2 or YOLO)')
parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
parser.add_argument('--min_score', type=float, default=0.50, help='Minimum score threshold (0-1)')
parser.add_argument('--yolo_model', type=str, default=None, help='YOLO model path/name (e.g., yolov8n.pt)')
parser.add_argument('--ocr', action='store_true', help='Enable OCR on selected snapshots (1-9 keys)')
args = parser.parse_args()

# Grab path to current working directory
CWD_PATH = os.getcwd()

# TF2 SavedModel
PATH_TO_SAVED_MODEL = os.path.join(CWD_PATH, MODEL_NAME, 'saved_model')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

category_index = parse_labelmap(PATH_TO_LABELS)

# Load TF2 SavedModel (serving_default)
use_yolo = args.yolo_model is not None
if not use_yolo:
    detect_module = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    infer = detect_module.signatures.get('serving_default', None)
    if infer is None:
        infer = next(iter(detect_module.signatures.values()))
    _, sig_kwargs = infer.structured_input_signature
    input_keys = list(sig_kwargs.keys())
    if not input_keys:
        raise RuntimeError('SavedModel signature has no inputs')
    input_key = input_keys[0]
else:
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)




def open_camera(index: int):
    cam = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cam.isOpened():
        cam = cv2.VideoCapture(index)
    cam.set(3,1280)
    cam.set(4,720)
    return cam

def close_camera(cam):
    try:
        cam.release()
    except Exception:
        pass

# Initialize webcam feed (prefer AVFoundation on macOS)
video = open_camera(args.camera)
is_paused = False
last_frame = None
ocr_text_lines = []  # last OCR output to display on panel

min_score_thresh = args.min_score
while True:

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    if not is_paused:
        ret, frame = video.read()
        if not ret:
            continue
        last_frame = frame.copy()
    else:
        # When paused, reuse last frame if available
        if last_frame is None:
            ret, frame = video.read()
            if not ret:
                continue
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()
    if not use_yolo:
        input_tensor = tf.convert_to_tensor(np.expand_dims(frame, axis=0), dtype=tf.uint8)
        outputs = infer(**{input_key: input_tensor})
        boxes = outputs['detection_boxes'][0].numpy()
        scores = outputs['detection_scores'][0].numpy()
        classes = outputs.get('detection_classes', None)
        if classes is not None:
            classes = classes[0].numpy().astype(np.int32)
        else:
            classes = np.ones((boxes.shape[0],), dtype=np.int32)
    else:
        res = yolo.predict(source=frame, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(np.int32)
            h, w = frame.shape[:2]
            boxes = []
            for x1, y1, x2, y2 in xyxy:
                boxes.append([y1 / h, x1 / w, y2 / h, x2 / w])
            boxes = np.array(boxes, dtype=np.float32)
            scores = conf
            classes = cls + 1
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int32)

    draw_boxes(
        frame,
        boxes,
        classes,
        scores,
        category_index,
        min_score=min_score_thresh,
        line_thickness=4,
    )

    # Build snapshot panel (top 9 detections by score)
    h, w = frame.shape[:2]
    dets = []
    for i in range(len(boxes)):
        sc = float(scores[i]) if scores is not None else 1.0
        if sc < min_score_thresh:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = max(0, int(xmin * w)), max(0, int(ymin * h))
        x2, y2 = min(w, int(xmax * w)), min(h, int(ymax * h))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2].copy()
        dets.append((sc, crop))
    dets.sort(key=lambda t: t[0], reverse=True)
    dets = dets[:9]

    panel = None
    thumb_w, thumb_h = 200, 120
    margin = 8
    if dets:
        tiles = []
        for idx, (sc, crop) in enumerate(dets, start=1):
            if crop.size == 0:
                continue
            t = cv2.resize(crop, (thumb_w, thumb_h))
            header = np.full((24, thumb_w, 3), 230, dtype=np.uint8)
            label = f"{idx}: {int(sc*100)}%"
            cv2.putText(header, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            tile = np.vstack([header, t])
            tiles.append(tile)
        if tiles:
            blocks = []
            for i, tile in enumerate(tiles):
                if i > 0:
                    blocks.append(np.full((margin, thumb_w, 3), 255, dtype=np.uint8))
                blocks.append(tile)
            panel = np.vstack(blocks)
    else:
        panel = np.full((thumb_h+24, thumb_w, 3), 245, dtype=np.uint8)
        cv2.putText(panel, 'No detections', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # Append OCR text area beneath thumbnails if available
    if ocr_text_lines:
        text_h = 28 + 22 * len(ocr_text_lines)
        text_canvas = np.full((text_h, panel.shape[1], 3), 250, dtype=np.uint8)
        cv2.putText(text_canvas, 'OCR Result:', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        y = 44
        for line in ocr_text_lines[:10]:
            cv2.putText(text_canvas, str(line), (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)
            y += 22
        panel = np.vstack([panel, np.full((margin, panel.shape[1], 3), 255, dtype=np.uint8), text_canvas])

    # Compose side-by-side view (resize panel to match frame height)
    if panel is not None:
        scale = frame.shape[0] / panel.shape[0]
        new_w = max(1, int(panel.shape[1] * scale))
        panel = cv2.resize(panel, (new_w, frame.shape[0]))
        composed = np.hstack([frame, panel])
        hud = composed
    else:
        hud = frame

    # HUD: controls helper
    cv2.putText(hud, 'q: quit  p: pause/resume  s: stop camera  b: start camera  1-9: OCR',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
    cv2.putText(hud, 'q: quit  p: pause/resume  s: stop camera  b: start camera  1-9: OCR',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    if is_paused:
        cv2.putText(hud, 'PAUSED', (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow('ID CARD DETECTOR', hud)

    key = cv2.waitKey(1) & 0xFF
    if args.ocr and dets and key in [ord(str(d)) for d in range(1, 10)]:
        try:
            sel = int(chr(key)) - 1
            _, crop = dets[sel]
            import easyocr
            reader = easyocr.Reader(['tr', 'en'], gpu=False)
            result = reader.readtext(crop, detail=0)
            print('\n[OCR {}]'.format(sel+1))
            for line in result:
                print(line)
            ocr_text_lines = result if isinstance(result, list) else [str(result)]
        except Exception as e:
            print('OCR error:', e)
            ocr_text_lines = ['OCR error: {}'.format(e)]
    # pause/resume
    if key == ord('p'):
        is_paused = not is_paused
    # stop camera
    if key == ord('s'):
        if video is not None:
            close_camera(video)
            video = None
            is_paused = True
    # (re)start camera
    if key == ord('b'):
        if video is None:
            video = open_camera(args.camera)
            is_paused = False
    if key == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

