import cv2
import torch
import socket
import time
import sys
import os
from pathlib import Path
import numpy as np


fps_list = []
frame_count = []


# === Add yolov5 to Python path ===
yolo_path = r"C:\Users\MY PC\Desktop\Me\My_Projects\S2\EOC2\yolov5"
sys.path.insert(0, yolo_path)

# === Now import properly ===
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

from utils.torch_utils import select_device
from utils.augmentations import letterbox

# === Load custom trained model ===
weights = os.path.join(yolo_path, "runs", "train", "eoc_exp150", "weights", "best.pt")
device = select_device('0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
print(f"Model loaded on {device} with stride {stride}")

# === Webcam ===
cap = cv2.VideoCapture(0)

# === TCP socket to Unity ===
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 5005))
server.listen(1)
print("Waiting for Unity to connect...")

conn, addr = server.accept()
print("Connected to:", addr)

prev_time = time.time()
c, sums = 0, 0

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords


def process_frame(frame):
    img = letterbox(frame, new_shape=640, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    if pred[0] is not None and len(pred[0]):
        pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], frame.shape).round()
    return pred[0]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    detections = process_frame(frame)

    detected_classes = set()

    if detections is not None and len(detections) > 0:
        for *xyxy, conf, cls in detections:
            class_name = names[int(cls)]
            detected_classes.add(class_name)

            label = f'{class_name} {conf:.2f}'
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # === Determine data to send to Unity ===
    if "Palm" in detected_classes:
        data_to_send = "1"
    else:
        data_to_send = "0"  # If Fist is present or nothing is detected

    conn.send(data_to_send.encode())
    print(f"Sent to Unity: {data_to_send}")

    # Calc FPS
    fps = 1 / (start_time - prev_time)
    c += 1
    sums += fps
    avg = sums / c
    print(f"FPS: {fps:.2f}, Avg: {avg:.2f}")
    prev_time = start_time
    
    fps_list.append(fps)
    frame_count.append(c)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('YOLO Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"The average FPS is {avg:.2f}")



import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(frame_count, fps_list, label='FPS per Frame', color='green')
plt.xlabel("Frame Index")
plt.ylabel("FPS")
plt.title("YOLOv5 Real-Time Inference FPS Over Time")
plt.grid(True)
plt.legend()
plt.savefig("fps_3splt.jpeg")  # Save the graph
print("plot saved")
plt.show()

cap.release()
cv2.destroyAllWindows()
conn.close()
server.close()
