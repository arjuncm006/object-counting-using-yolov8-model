



## Introduction

This project involves annotating a dataset containing images with boxes, training a YOLOv8 model for object detection, and using the trained model to detect boxes in images. The steps below detail the entire process from dataset preparation to model training and testing.

## Requirements

- **Python 3.x**
- **OpenCV 4.x**
- **Ultralytics**
- **Pytorch**
- **LabelImg**

## Software and Tools

- **Python:** Programming language used for the script.
- **OpenCV:** Library for computer vision tasks.
- **Ultralytics:** Ultralytics specializes in high-performance AI models, notably the YOLO object detection framework.
- **Pytorch:** PyTorch is an open-source deep learning framework for building and training neural networks.
- **LabelImg:** LabelImg is an open-source graphical image annotation tool used for labeling objects in images.

## Dataset Preparation
[Labeling image](https://drive.google.com/drive/folders/1WHpExY04EewfeeqJdMxsMg6maO7FCz8_?usp=drive_link)
1. **Annotate Images**: Use `labelimg` to annotate the dataset containing images with boxes. Save the annotations in the appropriate format.

2. **Split Dataset**: Split the annotated dataset into training and validation sets, each containing `images` and `labels` folders.

## Configuration File

Create a `data_custom.yaml` file with the following configuration:
```yaml
train: D:\box\train
val: D:\box\val
nc: 1
names: ["box"]
```

## Model Setup
1. Download YOLOv8 Weights: Download the yolov8.pt file from the Ultralytics GitHub repository.

2. Install PyTorch: Open a command prompt in the root directory of your project and install PyTorch using the following command: 
```bash
pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Install Ultralytics: Install the Ultralytics package using the command:
```bash
pip install ultralytics
```
4. Activate Virtual Environment: Activate your virtual environment.

5. Verify Installation: Check if PyTorch is installed and CUDA is available using the following commands:
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

## Training the Model
Train the YOLOv8 model using the following command:

```bash
yolo task=detect mode=train epochs=30 data=data_custom.yaml model=yolov8m.pt imgsz=640 batch=3
```
This process might take some time. Upon completion, you will get a best.pt file located inside runs>detect>train>weights.

### Rename Trained Weights
Rename the best.pt file to yolov8m_custom.pt and move it to the root directory.

### Model Inference
To detect boxes in an image using the trained model, use the following command:

```bash
yolo task=detect mode=predict model=yolov8m_custom.pt show=True conf=0.5 source=1.jpg
```
## Explanation Code 
### Python Code for Box Detection
Use the following Python code to import the model yolov8m_custom.pt and detect boxes in an image, displaying the number of boxes detected:



MY CODE

import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("C:/Users/harshini/OneDrive/Documents/GitHub/Counting-of-Objects-in-an-image/runs/detect/yolov8n_custom7/weights/best.pt")

# Set the confidence threshold
conf = 0.25

# Set the source image
source = "images (1).jpeg"

# Set save to True
save = True

# Perform object detection
results = model(source, conf=conf, save=save)


