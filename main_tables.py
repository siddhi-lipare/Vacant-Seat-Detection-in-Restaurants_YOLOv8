from ultralytics import YOLO
import cv2
import numpy as np
from iou import calculate_iou  

# Load the YOLO model
model = YOLO('yolov8l.pt')

# Load your image
image = cv2.imread('data/frame1.jpg')

# Detect objects in the image
results = model(image, show=True, conf=0.3)

classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Extract chair, person, and table boxes
chair_boxes = []
person_boxes = []
table_boxes = []

for r in results:
    boxes = r.boxes
    for box in boxes:
        c = int(box.cls[0])
        cc = classNames[c]
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if cc == "person":
            person_boxes.append((x1, y1, x2, y2))
        elif cc == "chair":
            chair_boxes.append((x1, y1, x2, y2))
        elif cc == "dining table":
            table_boxes.append((x1, y1, x2, y2))

unoccupied_chairs = []
for chair_box in chair_boxes:
    is_unoccupied = True

    for person_box in person_boxes:
        iou = calculate_iou(chair_box, person_box)

        if iou >= 0.2:
            is_unoccupied = False
            break

    if is_unoccupied:
        unoccupied_chairs.append(chair_box)

empty_tables = []
for table_box in table_boxes:
    is_empty = True

    for chair_box in unoccupied_chairs:
        iou = calculate_iou(table_box, chair_box)

        if iou >= 0.2:
            is_empty = False
            break

    if is_empty:
        empty_tables.append(table_box)
# Draw bounding boxes for empty tables (blue color)
for table_box in empty_tables:
    x1, y1, x2, y2 = table_box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 13)


cv2.putText(image, "Empty tables: {}".format(len(empty_tables)), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
# Save the output image as 'output.jpg'
cv2.imwrite('output_tables.jpg', image)

# Display the image with bounding boxes
cv2.imshow('Empty Tables Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
