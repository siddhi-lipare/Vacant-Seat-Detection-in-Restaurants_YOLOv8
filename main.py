from ultralytics import YOLO
import cv2
from iou import calculate_iou

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
results = model('data/frame1.jpg', show=True, conf=0.6)
image = cv2.imread('data/frame1.jpg')

person_boxes = []
chair_boxes = []

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

# Apply Intersection over Union
for chair_box in chair_boxes:
    for person_box in person_boxes:
        iou = calculate_iou(chair_box, person_box)
        
        # If IoU is above a threshold, change the chair's bounding box color to red
        if iou > 0.3:
            x1_chair, y1_chair, x2_chair, y2_chair = chair_box
            color = (0, 0, 255)  # Red color for chair
            thickness = 13  # Line thickness
            cv2.rectangle(image, (x1_chair, y1_chair), (x2_chair, y2_chair), color, thickness)
            # Put text on the image
            cv2.putText(image, "Occupied", (x1_chair, y1_chair - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)

# Draw bounding boxes for persons (green color)
for person_box in person_boxes:
    x1_person, y1_person, x2_person, y2_person = person_box
    color = (0, 255, 0)  # Green color for persons
    thickness = 5  # Line thickness
    cv2.rectangle(image, (x1_person, y1_person), (x2_person, y2_person), color, thickness)

cv2.namedWindow("Image with Bounding Boxes", cv2.WINDOW_NORMAL)
cv2.imshow("Image with Bounding Boxes", image)
cv2.imwrite('output.jpg', image)

# Define a flag to control the loop
exit_flag = False

while True:
    key = cv2.waitKey(1) & 0xFF  # Get the key pressed
    if key == ord('q'):
        exit_flag = True  # Set the exit flag to True and break out of the loop
        break

if exit_flag:
    cv2.destroyAllWindows()  # Close the OpenCV window
