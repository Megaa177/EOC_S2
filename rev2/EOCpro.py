import cv2
import torch
import socket
import time

# Load YOLO model (make sure to download YOLOv5 from ultralytics)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(device)

# Setup webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Setup TCP server for Unity communication
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 5005))  # Bind to localhost on port 5005
server.listen(1)
print("Waiting for Unity to connect...")

conn, addr = server.accept()
print("Connected to:", addr)

prev_time = time.time()  # Initialize for FPS calculation

def process_frame(frame):
    """ Process frame with YOLO """
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get detections as pandas DataFrame
    return detections


c=0
sums=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()  # Time at start of frame processing

    detections = process_frame(frame)

    # Example logic: If person is detected, send "1" else "0"
    data_to_send = "1" if 'person' in detections['name'].values else "0"
    conn.send(data_to_send.encode())  # Send detection result to Unity

    # Calculate FPS
    fps = 1 / (start_time - prev_time)
    c+=1
    sums=sums+fps
    print(fps)
    avg=sums/c
    print(avg," is avg")
    prev_time = start_time
    

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('YOLO Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
conn.close()
server.close()
