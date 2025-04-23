import os
import sys

# Add yolov5 to Python path
yolo_path = os.path.abspath(os.path.join("..", "yolov5"))
sys.path.append(yolo_path)

from train import run  # This is from yolov5's train.py

# --- CONFIG ---
DATA_YAML = "data.yaml"       # must be in this same folder
MODEL_CFG = "yolov5s.yaml"    # model architecture
PRETRAINED = "yolov5s.pt"     # pre-trained weights to start from
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
RUN_NAME = "eoc_exp"          # folder name under runs/train/
DEVICE = 0                    # Use GPU 0

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    run(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        epochs=EPOCHS,
        weights=PRETRAINED,
        cfg=MODEL_CFG,
        name=RUN_NAME,
        device=DEVICE
    )
