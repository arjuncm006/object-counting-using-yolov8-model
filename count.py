
from ultralytics import YOLO

# Load the  pretrained YOLO model
model = YOLO("C:/Users/harshini/OneDrive/Documents/GitHub/Counting-of-Objects-in-an-image/runs/detect/yolov8n_custom7/weights/best.pt")

# Set the confidence threshold
conf = 0.25

# Set the source image
source = "images (1).jpeg"

# Set save to True
save = True

# Perform object detection
results = model(source, conf=conf, save=save)

