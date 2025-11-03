
# Import packages
import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import sys

from PIL import Image

# This project originally used TF Object Detection API utils; we replace them with
# lightweight helpers to remove the heavy dependency.

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
                # lines like: name: 'idcard'
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
    # Fallback
    if not classes:
        classes = {1: {'id': 1, 'name': 'object'}}
    return classes

def draw_boxes(image, boxes, classes, scores, category_index, min_score, line_thickness=3):
    h, w = image.shape[:2]
    best_idx = None
    if scores is not None and len(scores) > 0:
        best_idx = int(np.argmax(scores))
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
    # Return top-scoring box coordinates for cropping
    if best_idx is not None and (scores[best_idx] >= min_score):
        ymin, xmin, ymax, xmax = boxes[best_idx]
        return [ymin, xmin, ymax, xmax]
    return [0.0, 0.0, 1.0, 1.0]

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'model'

# CLI args (modern)
parser = argparse.ArgumentParser(description='ID Card Detection (TF2 SavedModel + optional OCR)')
parser.add_argument('--image', type=str, default=None, help='Absolute or relative path to image file')
parser.add_argument('--min_score', type=float, default=0.60, help='Minimum score threshold (0-1)')
parser.add_argument('--ocr', action='store_true', help='Run OCR on cropped ROI')
args = parser.parse_args()

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Prefer TF2 SavedModel
PATH_TO_SAVED_MODEL = os.path.join(CWD_PATH, MODEL_NAME, 'saved_model')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','labelmap.pbtxt')
CATEGORY_INDEX = parse_labelmap(PATH_TO_LABELS)

# Path to image
if args.image is not None:
    PATH_TO_IMAGE = args.image if os.path.isabs(args.image) else os.path.join(CWD_PATH, args.image)
else:
    IMAGE_NAME = 'test_images/image1.png'
    PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Label map already parsed into CATEGORY_INDEX

# Load TF2 SavedModel
detect_module = tf.saved_model.load(PATH_TO_SAVED_MODEL)
infer = detect_module.signatures.get('serving_default', None)
if infer is None:
    # Fallback: ilk imzayı kullan
    infer = next(iter(detect_module.signatures.values()))

# Define input and run detection with TF2

# Load image and run detection (TF2)
image = cv2.imread(PATH_TO_IMAGE)
if image is None:
    raise FileNotFoundError('Image not found: {}'.format(PATH_TO_IMAGE))
input_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.uint8)

# İmzanın beklediği girdi anahtarını tespit et
_, sig_kwargs = infer.structured_input_signature
input_keys = list(sig_kwargs.keys())
if not input_keys:
    raise RuntimeError('SavedModel signature has no inputs')
input_key = input_keys[0]

outputs = infer(**{input_key: input_tensor})
boxes = outputs['detection_boxes'][0].numpy()
scores = outputs['detection_scores'][0].numpy()
classes = outputs.get('detection_classes', None)
if classes is not None:
    classes = classes[0].numpy().astype(np.int32)
else:
    classes = np.ones((boxes.shape[0],), dtype=np.int32)

min_thresh = args.min_score
array_coord = draw_boxes(
    image,
    boxes,
    classes,
    scores,
    CATEGORY_INDEX,
    min_score=min_thresh,
    line_thickness=3,
)

ymin, xmin, ymax, xmax = array_coord

shape = np.shape(image)
im_width, im_height = shape[1], shape[0]
(left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))

# Using Image to crop and save the extracted copied image
output_path = os.path.join(CWD_PATH, 'output_cropped.png')
im = Image.open(PATH_TO_IMAGE)
roi_box = (max(0, left), max(0, top), min(im_width, right), min(im_height, bottom))
im.crop(roi_box).save(output_path, quality=95)

# Optional OCR on cropped ROI
if args.ocr:
    try:
        import easyocr
        reader = easyocr.Reader(['tr', 'en'], gpu=False)
        ocr_img = np.array(Image.open(output_path))
        ocr_result = reader.readtext(ocr_img, detail=0)
        print('\nOCR output:')
        for line in ocr_result:
            print(line)
    except Exception as e:
        print('OCR failed: {}'.format(e))

# Show results
cv2.imshow('ID CARD DETECTOR', image)

image_cropped = cv2.imread(output_path)
cv2.imshow("ID-CARD-CROPPED", image_cropped)

# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
