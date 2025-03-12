# Install Dependencies
!pip install albumentations

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import zipfile

# Extract dataset from Google Drive
dataset_zip = "/content/drive/MyDrive/Data_Eoc.zip"
extract_path = "/content/Data_Eoc"

if not os.path.exists(dataset_zip):
    raise FileNotFoundError(f"Dataset not found at {dataset_zip}")

with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall("/content/")

# Clone YOLOv5 Repository
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt

# Disable wandb
import os
os.environ['WANDB_DISABLED'] = 'true'


# Train YOLOv5
!python train.py --img 640 --batch 16 --epochs 50 --data /content/Data_Eoc/data.yaml --weights yolov5s.pt --cache

# Export Model
!python export.py --weights runs/train/exp/weights/best.pt --include onnx

# Save Model to Google Drive
trained_model_path = "/content/yolov5/runs/train/exp/weights/best.pt"
destination_path = "/content/drive/MyDrive/best.pt"

if os.path.exists(trained_model_path):
    !cp {trained_model_path} {destination_path}
    print(f"Model saved to {destination_path}")
else:
    print("Training failed or model file not found.")
