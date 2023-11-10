import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from iou import calculate_iou

model = YOLO('yolov8l.pt')

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

# Input and Output video file paths
input_video_path = 'data/ich.mp4'
output_video_path = 'output_video_time.mp4'
csv_file = 'empty_tables_data.csv'

# Open the video file
cap = cv2.VideoCapture(input_video_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('outputvid_tables.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Initialize a list to store rows
rows = []

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects in the frame
    results = model(frame, show=False, conf=0.6)

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
        is_empty = True

        for person_box in person_boxes:
            iou = calculate_iou(chair_box, person_box)

            if iou >= 0.3:
                is_empty = False
                break

        if is_empty:
            unoccupied_chairs.append(chair_box)

    empty_tables = []
    for table_box in table_boxes:
        is_empty = True

        for chair_box in unoccupied_chairs:
            iou = calculate_iou(table_box, chair_box)

            if iou >= 0.3:
                is_empty = False
                break

        if is_empty:
            empty_tables.append(table_box)

    current_time = datetime.now().strftime('%H:%M:%S')
    empty_table_count = len(empty_tables)
    rows.append({'Timestamp': current_time, 'Empty Tables Count': empty_table_count})

    for table_box in empty_tables:
        x1, y1, x2, y2 = table_box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

    cv2.putText(frame, "Empty tables: {}".format(len(empty_tables)), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)

    cv2.imshow('Empty Tables Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Create a DataFrame from the rows list
df = pd.DataFrame(rows)

# Write DataFrame to CSV file
df.to_csv(csv_file, index=False)

# Release everything when finished
cap.release()
out.release()
cv2.destroyAllWindows()
