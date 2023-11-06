from ultralytics import YOLO
import cv2
import numpy as np

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
model = YOLO('yolov8l.pt')
results = model('data/frame1.jpg', show=True, conf=0.80)
image = cv2.imread('data/frame1.jpg')

for r in results:
    boxes = r.boxes
    for box in boxes:
        c = int(box.cls[0])
        cc = classNames[c]
        if cc in ["person", "chair"]:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box on the image
            color = (0, 255, 0)  # Green color
            thickness = 13  # Line thickness
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

cv2.namedWindow("Image with Bounding Boxes", cv2.WINDOW_NORMAL)
cv2.imshow("Image with Bounding Boxes", image)

# Define a flag to control the loop
exit_flag = False

while True:
    key = cv2.waitKey(1) & 0xFF  # Get the key pressed
    if key == ord('q'):
        exit_flag = True  # Set the exit flag to True and break out of the loop
        break

if exit_flag:
    cv2.destroyAllWindows()  # Close the OpenCV window

cv2.imwrite('output.jpg', image)  # Save the image with bounding boxes