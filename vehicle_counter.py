import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# ============================
# Load YOLO Model (Pretrained)
# ============================
model = YOLO("yolov8n.pt")

# ============================
# Video Input / Output
# ============================
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "output_.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

# ============================
# ROI Line
# ============================
roi_y = int(frame_height * 0.6)

# ============================
# Tracking Variables
# ============================
counted_ids = set()
car_count = 0
motorcycle_count = 0
pedestrian_count = 0

results_list = []

CLASS_NAMES = {0: "pedestrian", 2: "car", 3: "motorcycle"}

frame_number = 0

# ============================
# Main Loop
# ============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # Detection + tracking
    results = model.track(frame, persist=True, classes=[0,2,3], conf=0.4)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            cls = int(cls)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            label = CLASS_NAMES.get(cls, "object")

            # Bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

            # ROI crossing
            if cy > roi_y and track_id not in counted_ids:
                counted_ids.add(track_id)
                if cls == 2: car_count += 1
                elif cls == 3: motorcycle_count += 1
                elif cls == 0: pedestrian_count += 1

                # Save detailed record
                results_list.append({
                    "Frame": frame_number,
                    "ID": track_id,
                    "Class": label,
                    "Center_X": cx,
                    "Center_Y": cy,
                    "BBox_X1": x1,
                    "BBox_Y1": y1,
                    "BBox_X2": x2,
                    "BBox_Y2": y2
                })

    # Draw ROI
    cv2.line(frame, (0, roi_y), (frame_width, roi_y), (255,0,0), 3)

    # Live counters
    cv2.putText(frame, f"Cars: {car_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Motorcycles: {motorcycle_count}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f"Pedestrians: {pedestrian_count}", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    out.write(frame)

# ============================
# Save CSV
# ============================
df = pd.DataFrame(results_list)
df.to_csv("results.csv", index=False)

# Release resources
cap.release()
out.release()

print("Processing Complete!")
print("Saved: output.mp4")
print("Saved: results.csv")