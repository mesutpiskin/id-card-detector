
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
parser = argparse.ArgumentParser(description='ID Card Detection (camera)')
parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
parser.add_argument('--min_score', type=float, default=0.50, help='Minimum score threshold (0-1)')
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
detect_module = tf.saved_model.load(PATH_TO_SAVED_MODEL)
infer = detect_module.signatures.get('serving_default', None)
if infer is None:
    infer = next(iter(detect_module.signatures.values()))


_, sig_kwargs = infer.structured_input_signature
input_keys = list(sig_kwargs.keys())
if not input_keys:
    raise RuntimeError('SavedModel signature has no inputs')
input_key = input_keys[0]

# Initialize webcam feed (prefer AVFoundation on macOS)
video = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
if not video.isOpened():
    video = cv2.VideoCapture(args.camera)
video.set(3,1280)
video.set(4,720)

min_score_thresh = args.min_score
while True:

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    if not ret:
        continue
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, axis=0), dtype=tf.uint8)
    outputs = infer(**{input_key: input_tensor})
    boxes = outputs['detection_boxes'][0].numpy()
    scores = outputs['detection_scores'][0].numpy()
    classes = outputs.get('detection_classes', None)
    if classes is not None:
        classes = classes[0].numpy().astype(np.int32)
    else:
        classes = np.ones((boxes.shape[0],), dtype=np.int32)

    draw_boxes(
        frame,
        boxes,
        classes,
        scores,
        category_index,
        min_score=min_score_thresh,
        line_thickness=4,
    )

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('ID CARD DETECTOR', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

