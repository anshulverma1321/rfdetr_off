import json
import os

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

add_markdown("# RF-DETR Complete Local Inference Pipeline\nThis notebook demonstrates a modular, production-style setup for running and evaluating RF-DETR models locally.")

add_markdown("## 1. Environment Verification\nCheck CUDA, PyTorch, and required libraries.")
add_code("""import sys
import os
sys.path.append('..') # Ensure we can import from utils

import torch
import cv2
import supervision as sv

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"Supervision Version: {sv.__version__}")""")

add_markdown("## 2. Directory Initialization\nAutomatically create required folders: datasets, models, logs, outputs.")
add_code("""directories = [
    "../datasets/input_images",
    "../datasets/input_videos",
    "../datasets/test_images",
    "../datasets/outputs/images",
    "../datasets/outputs/videos",
    "../datasets/outputs/metrics",
    "../models/pretrained",
    "../logs"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Verified directory: {directory}")""")

add_markdown("## 3. Load Pretrained RF-DETR Model\nInitialize the model using our wrapper. This handles offline caching and device mapping.")
add_code("""from utils.inference import RFDETRInference

# Initialize the Medium model. Change size="N", "S", "L", "XL" as needed.
model_wrapper = RFDETRInference(size="M", config_path="../configs/config.json")
print("Model loaded successfully.")""")

add_markdown("## 4. Visualization & 5. FPS Metrics Setup\nImport visualization and performance tracking utilities.")
add_code("""from utils.visualization import Visualizer
from utils.fps import PerformanceTracker

visualizer = Visualizer()
tracker = PerformanceTracker()""")

add_markdown("## 6. Image Inference Pipeline\nProcess a single image, track inference time, and save the result.")
add_code("""from PIL import Image
import numpy as np

# Create a dummy image for validation if none exists
dummy_image_path = "../datasets/input_images/sample.jpg"
if not os.path.exists(dummy_image_path):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Sample Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imwrite(dummy_image_path, img)

image = cv2.imread(dummy_image_path)

tracker.start()
detections = model_wrapper.predict(image, threshold=0.5)
tracker.stop()

# Visualize
annotated_image = visualizer.annotate(image, detections)

# Save output
output_path = "../datasets/outputs/images/sample_output.jpg"
cv2.imwrite(output_path, annotated_image)

print(f"Saved annotated image to {output_path}")
tracker.log_metrics()""")

add_markdown("## 7. Video Inference Pipeline\nRun frame-by-frame inference on a video and generate an annotated output video.")
add_code("""from utils.video_utils import VideoProcessor

video_input = "../datasets/input_videos/sample.mp4"
video_output = "../datasets/outputs/videos/sample_output.mp4"

# Create a dummy video for validation if none exists
if not os.path.exists(video_input):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_input, fourcc, 30.0, (640, 480))
    for i in range(30):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (100 + i*10, 240), 50, (0, 0, 255), -1)
        out.write(img)
    out.release()

processor = VideoProcessor(video_input)

# Define callbacks
def process_frame(frame):
    return model_wrapper.predict(frame, threshold=0.5)

def visualize_frame(frame, detections):
    return visualizer.annotate(frame, detections)

# Note: Set output_path to None if you just want to run without saving
processor.process_video(
    inference_callback=process_frame,
    visualization_callback=visualize_frame,
    output_path=video_output
)""")

add_markdown("## 8. Webcam / Live Camera Inference\nRead from webcam, perform real-time detection, and display metrics.")
add_code("""# Note: Set webcam_id to 0 for default camera.
# This cell is commented out to prevent CI/CD or headless environment hangs.

'''
webcam_id = 0
cap = cv2.VideoCapture(webcam_id)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    tracker.start()
    detections = model_wrapper.predict(frame, threshold=0.5)
    tracker.stop()
    
    annotated_frame = visualizer.annotate(frame, detections)
    
    # Render FPS
    metrics = tracker.get_metrics()
    cv2.putText(annotated_frame, f"FPS: {metrics['fps']:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("RF-DETR Live", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
'''
print("Webcam inference code is ready. Uncomment to run locally.")""")

add_markdown("## 9. Evaluation Metrics\nInitialize the COCO evaluation framework for validation on test datasets.")
add_code("""from utils.metrics import COCOEvaluator

ground_truth_json = "../datasets/test_images/annotations.json"
evaluator = COCOEvaluator(ground_truth_json)

# Note: Once you have a populated dataset, you can loop through images,
# gather `model_predictions` dict mapping image_name -> sv.Detections
# and run `evaluator.evaluate(model_predictions)`.
print("Evaluation framework initialized.")""")

add_markdown("## 10. Logging System\nSave metrics and inference summaries to the logs directory.")
add_code("""import json
import datetime

log_data = {
    "timestamp": str(datetime.datetime.now()),
    "model_size": model_wrapper.size,
    "performance": tracker.get_metrics()
}

log_path = "../logs/execution_summary.json"
with open(log_path, "w") as f:
    json.dump(log_data, f, indent=4)
    
print(f"Logged execution summary to {log_path}")""")

add_markdown("## 11. Error Handling\nDemonstrate robust handling of missing files and invalid inputs.")
add_code("""try:
    print("Testing error handling on missing image...")
    # Passing None to simulate failed imread
    detections = model_wrapper.predict(None)
    if detections is None:
        print("Model handled invalid input safely (returned None).")
except Exception as e:
    print(f"Caught exception: {e}")""")

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/rfdetr_pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Notebook created successfully.")
