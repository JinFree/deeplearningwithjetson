import os

IMAGE_ROOT = "/workspace/images"
VIDEO_PATH = "/workspace/challenge.mp4"
IMAGE_PATH_LIST = [os.path.join(IMAGE_ROOT, "bus.jpg"), os.path.join(IMAGE_ROOT, "zidane.jpg")]


def inference_image_with_ultralytics(model_path, image_path_list, save=False):
    from ultralytics import YOLO
    model = YOLO(model_path)
    for image_path in image_path_list:
        model(image_path, save=save)
    return


def inference_video_with_ultralytics(model_path, video_path, save=False):
    from ultralytics import YOLO
    model = YOLO(model_path)
    model(video_path, save=save)
    return


def convert_to_onnx(model_name, input_size=640):
    from ultralytics import YOLO
    model = YOLO(model_name)
    model.export(format='onnx', imgsz=input_size)
    os.system(f'mv {model_name.replace(".pt", ".onnx")} {model_name.replace(".pt", "")}_{input_size}.onnx')
    return 


def convert_to_engine(model_name, input_size=640, half=False):
    from ultralytics import YOLO
    model = YOLO(model_name)
    model.export(format='engine', imgsz=input_size, half=half)
    os.system(f'mv {model_name.replace(".pt", ".engine")} {model_name.replace(".pt", "")}_fp{"16" if half else "32"}_{input_size}.engine')
    return


if __name__ == "__main__":
    # Ultralytics docker 환경에서 YOLO11n 모델과 YOLO12n 모델을 onnx로 변환합니다.
    convert_to_onnx('yolo11n.pt', 640)
    convert_to_onnx('yolo12n.pt', 640)
    
    # YOLO11n onnx로 이미지 추론 시간을 확인합니다.
    inference_image_with_ultralytics('yolo11n_640.onnx', IMAGE_PATH_LIST, save=True)
    
    # YOLO12n onnx로 이미지 추론 시간을 확인합니다.
    inference_image_with_ultralytics('yolo12n_640.onnx', IMAGE_PATH_LIST, save=True)
    
    # YOLO11n 모델을 TensorRT로로 변환합니다. 이 때, fp32와 fp16 두 종류로 합니다.
    convert_to_engine('yolo11n.pt', 640, half=False)
    convert_to_engine('yolo11n.pt', 640, half=True)
    
    # TensorRT 변환한 YOLO11n 모델을 추론하여 속도를 확인합니다.
    inference_video_with_ultralytics('yolo11n_fp32_640.engine', VIDEO_PATH, save=True)
    inference_video_with_ultralytics('yolo11n_fp16_640.engine', VIDEO_PATH, save=True)

    # YOLO11n 모델을 다른 해상도로 TensorRT변환하여 속도를 확인합니다.
    convert_to_engine('yolo11n.pt', 416, half=True)
    inference_video_with_ultralytics('yolo11n_fp16_416.engine', VIDEO_PATH, save=True)
    
    # YOLO12n 모델을 같은 방식으로 변환하여 속도를 확인합니다.
    convert_to_engine('yolo12n.pt', 416, half=True)
    inference_video_with_ultralytics('yolo12n_fp16_416.engine', VIDEO_PATH, save=True)
    
    # YOLOv8n 모델을 같은 방식으로 변환하여 속도를 확인합니다.
    convert_to_engine('yolov8n.pt', 416, half=True)
    inference_video_with_ultralytics('yolov8n_fp16_416.engine', VIDEO_PATH, save=True)
    