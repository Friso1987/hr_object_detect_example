import cv2
import numpy as np
import tensorflow as tf
import re
import os
import csv
from datetime import datetime

# Function to parse label map file and return a dictionary of id to display_name
def load_labelmap(labelmap_path):
    labelmap_dict = {}
    with open(labelmap_path, 'r') as file:
        content = file.read()
        items = re.findall(r'item {\s*name: "(.*?)"\s*id: (\d+)\s*display_name: "(.*?)"\s*}', content)
        for item in items:
            labelmap_dict[int(item[1])] = item[2]
    return labelmap_dict

# Function to log detections to a CSV file
def log_detections(csv_path, detection_time, detections_summary):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header only if the file is being created for the first time
            writer.writerow(['Datetime', 'Object', 'Count'])
        for label, count in detections_summary.items():
            writer.writerow([detection_time, label, count])

# Load the pre-trained model
model = tf.saved_model.load('ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model')

# Load the label map and create a dictionary
labelmap_dict = load_labelmap('labelmap.pbtxt')

# Path to the CSV file
csv_path = 'object_detections.csv'

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor without changing the dtype
    input_tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)

    # Run object detection
    detections = model(input_tensor)

    # Extract detection results
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # Prepare to log detected objects
    detections_summary = {}

    # Draw bounding boxes and labels
    for i in range(num_detections):
        box = detections['detection_boxes'][i]
        score = detections['detection_scores'][i]
        label_id = detections['detection_classes'][i].astype(int)
        label = labelmap_dict.get(label_id, "Unknown")

        if score > 0.5:
            if label in detections_summary:
                detections_summary[label] += 1
            else:
                detections_summary[label] = 1
            
            ymin, xmin, ymax, xmax = box
            start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
            end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))

            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            cv2.putText(frame, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Log the detections after processing each frame
    detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_detections(csv_path, detection_time, detections_summary)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
