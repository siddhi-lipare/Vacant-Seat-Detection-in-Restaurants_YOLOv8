import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
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
video_path = "data/sudarshan2.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Define the video writer to save the output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('out/outputvid.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Initialize empty seats count and list for data collection
empty_seats_count = []
timestamps = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame, conf=0.5)

    # Extract and categorize bounding boxes
    occupied_chairs = []
    empty_count = 0
    person_boxes = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = int(box.cls[0])
            cc = classNames[c]
            x1, y1, x2, y2 = box.xyxy[0]

            if cc == "chair":
                is_occupied = False

                for person_box in person_boxes:
                    # Calculate IoU with previously detected persons
                    iou = calculate_iou((x1, y1, x2, y2), person_box)
                    if iou >= 0.2:
                        is_occupied = True
                        break

                if is_occupied:
                    color = (0, 0, 255)  # Red color for occupied chair
                    text = "Occupied"
                else:
                    empty_count += 1
                    color = (0, 255, 0)  # Green color for unoccupied chair
                    text = "Empty"

                thickness = 13 if is_occupied else 6  # Line thickness
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                occupied_chairs.append((x1, y1, x2, y2))

                # Write text on the bounding box
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            elif cc == "person":
                person_boxes.append((x1, y1, x2, y2))
                color = (255, 0, 255)  # Purple color for persons
                thickness = 2  # Line thickness
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    # Add a text to display the count of empty chairs
    cv2.putText(frame, "Empty chairs: {}".format(empty_count), (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    empty_seats_count.append(empty_count)
    timestamps.append(datetime.now())  # Record current timestamp

    output_video.write(frame)  # Write the frame with bounding boxes to the output video

    cv2.imshow("Video with Bounding Boxes", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video objects
cap.release()
output_video.release()
cv2.destroyAllWindows()

# Create a DataFrame to store empty seats count and timestamps
data = {'Timestamp': timestamps, 'Empty Seats Count': empty_seats_count}
df = pd.DataFrame(data)

# Save data to CSV file
df.to_csv('csv/empty_seats_data.csv', index=False)
