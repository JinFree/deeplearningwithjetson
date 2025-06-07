from ultralytics import YOLO
import os
img_root = "/workspace/images"
video_path = "/workspace/challenge.mp4"

# Ultralytics docker 환경에서 YOLO11n 모델과 YOLO12n 모델을 onnx로 변환합니다.
model_yolo11n = YOLO('yolo11n.pt')
model_yolo11n.export(format='onnx')

model_yolo12n = YOLO('yolo12n.pt')
model_yolo12n.export(format='onnx')

# YOLO11n onnx로 이미지 추론 시간을 확인합니다.
model_yolo11n = YOLO('yolo11n.onnx')
model_yolo11n(os.path.join(img_root, "bus.jpg"), save=True)
model_yolo11n(os.path.join(img_root, "zidane.jpg"), save=True)

# YOLO12n onnx로 이미지 추론 시간을 확인합니다.
model_yolo12n = YOLO('yolo12n.onnx')
model_yolo12n(os.path.join(img_root, "bus.jpg"), save=True)
model_yolo12n(os.path.join(img_root, "zidane.jpg"), save=True)

# YOLO11n 모델을 TensorRT로로 변환합니다. 이 때, fp32와 fp16 두 종류로 합니다.
model_yolo11n = YOLO('yolo11n.pt')
model_yolo11n.export(format='engine')
os.system("mv yolo11n.engine yolo11n_fp32.engine")

model_yolo11n.export(format='engine', half=True)
os.system("mv yolo11n.engine yolo11n_fp16.engine")

# TensorRT 변환한 YOLO11n 모델을 추론하여 속도를 확인합니다.
model_yolo11n = YOLO('yolo11n_fp32.engine')
model_yolo11n(video_path, save=False)

model_yolo11n = YOLO('yolo11n_fp16.engine')
model_yolo11n(video_path, save=False)

# YOLO11n 모델을 다른 해상도로 TensorRT변환하여 속도를 확인합니다.
model_yolo11n = YOLO('yolo11n.pt')
model_yolo11n.export(format='onnx', imgsz=416)
os.system("mv yolo11n.onnx yolo11n_416.onnx")

model_yolo11n.export(format='engine', half=True, imgsz=416)
os.system("mv yolo11n.engine yolo11n_fp16_416.engine")

model_yolo11n = YOLO('yolo11n_fp16_416.engine')
model_yolo11n(video_path, save=True)

# YOLO12n 모델을 같은 방식으로 변환하여 속도를 확인합니다.
model_yolo12n = YOLO('yolo12n.pt')

model_yolo12n.export(format='onnx', imgsz=416)
os.system("mv yolo12n.onnx yolo12n_416.onnx")

model_yolo12n.export(format='engine', half=True, imgsz=416)
os.system("mv yolo12n.engine yolo12n_fp16_416.engine")

model_yolo12n = YOLO('yolo12n_fp16_416.engine')
model_yolo12n(video_path, save=True)

# YOLOv8n 모델을 같은 방식으로 변환하여 속도를 확인합니다.

model_yolov8n = YOLO('yolov8n.pt')

model_yolov8n.export(format='onnx', imgsz=416)
os.system("mv yolov8n.onnx yolov8n_416.onnx")

model_yolov8n.export(format='engine', half=True, imgsz=416)
os.system("mv yolov8n.engine yolov8n_fp16_416.engine")

model_yolov8n = YOLO('yolov8n_fp16_416.engine')
model_yolov8n(video_path, save=True)