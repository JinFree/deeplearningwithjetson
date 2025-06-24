Jetson-containers의 ML 이미지를 실행합니다.
```
docker run -it --privileged --rm --ipc=host --network=host --runtime=nvidia -v /home/nvidia/workspace:/workspace dustynv/l4t-ml:r32.7.1
```
- 이후의 실습에서 파일이 새로 써지지 않는 권한 문제가 발생하는 경우 새 터미널에서 아래 명령어를 입력합니다.
  ```
  sudo chown -R nvidia ~/workspace
  ```
- 아래와 같은 로그를 확인하면 JupyterLab에 접근할 수 있게 됩니다.
  ```
  allow 10 sec for JupyterLab to start @ http://IP:8888 (password nvidia)
  JupterLab logging location:  /var/log/jupyter.log  (inside the container)
  ```

Ultralytics YOLO onnx 파일로 이미지 추론을 수행합니다.
```
cd /workspace/files_from_ultralytics
python3 /workspace/deeplearningwithjetson/Object_Detection_with_Jetson_Nano/4_Ultralytics_inference_onnx.py \
  --model /workspace/files_from_ultralytics/yolov8n_416.onnx \
  --image /workspace/images/bus.jpg \
  --input_width 416 \
  --conf_thresh 0.2 \
  --nms_thresh 0.3
python3 /workspace/deeplearningwithjetson/Object_Detection_with_Jetson_Nano/4_Ultralytics_inference_onnx.py \
  --model /workspace/files_from_ultralytics/yolo11n_416.onnx \
  --image /workspace/images/bus.jpg \
  --input_width 416 \
  --conf_thresh 0.2 \
  --nms_thresh 0.3
python3 /workspace/deeplearningwithjetson/Object_Detection_with_Jetson_Nano/4_Ultralytics_inference_onnx.py \
  --model /workspace/files_from_ultralytics/yolo12n_416.onnx \
  --image /workspace/images/bus.jpg \
  --input_width 416 \
  --conf_thresh 0.2 \
  --nms_thresh 0.3
```

Ultralytics YOLO engine 파일로 이미지 추론을 수행합니다.
```
cd /workspace/files_from_ultralytics
python3 /workspace/deeplearningwithjetson/Object_Detection_with_Jetson_Nano/4_Ultralytics_inference_tensorrt.py \
  --model /workspace/files_from_ultralytics/yolov8n_fp16_416.engine \
  --image /workspace/images/bus.jpg \
  --input_width 416 \
  --conf_thresh 0.2 \
  --nms_thresh 0.3
```
- 아래와 같은 오류가 발생하는 것을 볼 수 있습니다.
  ```
  [06/07/2025-06:36:09] [TRT] [E] 1: [stdArchiveReader.cpp::StdArchiveReader::30] Error Code 1: Serialization (Serialization assertion magicTagRead == magicTag failed.Magic tag does not match)
  [06/07/2025-06:36:09] [TRT] [E] 4: [runtime.cpp::deserializeCudaEngine::50] Error Code 4: Internal Error (Engine deserialization failed.)
  ```
- Ultralytics docker의 TensorRT 버전과 ML docker의 TensorRT 버전이 다르기 때문에 발생하는 오류로, TensorRT 엔진은 같은 GPU의 같은 CUDA, CuDNN, TensorRT 버전에서 빌드한 것만 사용할 수 있음을 확인할 수 있습니다.


모든 도커 컨테이너를 종료하고, onnx 파일이 있는 경로의 권한을 변경합니다.
```
sudo chown -R nvidia ~/workspace/files_from_ultralytics
```

TensorRT를 이용해 onnx를 trt engine으로 빌드합니다.
```
trtexec \
  --onnx=/home/nvidia/workspace/files_from_ultralytics/yolov8n_416.onnx \
  --saveEngine=/home/nvidia/workspace/files_from_ultralytics/yolov8n_fp16_416_host.engine \
  --fp16
trtexec \
  --onnx=/home/nvidia/workspace/files_from_ultralytics/yolo11n_416.onnx \
  --saveEngine=/home/nvidia/workspace/files_from_ultralytics/yolo11n_fp16_416_host.engine \
  --fp16
trtexec \
  --onnx=/home/nvidia/workspace/files_from_ultralytics/yolo12n_416.onnx \
  --saveEngine=/home/nvidia/workspace/files_from_ultralytics/yolo12n_fp16_416_host.engine \
  --fp16
```

빌드한 trt engine으로 이미지 추론을 수행합니다.
```
python3 /home/nvidia/workspace/deeplearningwithjetson/Object_Detection_with_Jetson_Nano/4_Ultralytics_inference_tensorrt.py \
  --model /home/nvidia/workspace/files_from_ultralytics/yolov8n_fp16_416_host.engine \
  --image /home/nvidia/workspace/images/bus.jpg \
  --input_width 416 \
  --conf_thresh 0.2 \
  --nms_thresh 0.3

python3 /home/nvidia/workspace/deeplearningwithjetson/Object_Detection_with_Jetson_Nano/4_Ultralytics_inference_tensorrt.py \
  --model /home/nvidia/workspace/files_from_ultralytics/yolo11n_fp16_416_host.engine \
  --image /home/nvidia/workspace/images/bus.jpg \
  --input_width 416 \
  --conf_thresh 0.2 \
  --nms_thresh 0.3

python3 /home/nvidia/workspace/deeplearningwithjetson/Object_Detection_with_Jetson_Nano/4_Ultralytics_inference_tensorrt.py \
  --model /home/nvidia/workspace/files_from_ultralytics/yolo12n_fp16_416_host.engine \
  --image /home/nvidia/workspace/images/bus.jpg \
  --input_width 416 \
  --conf_thresh 0.2 \
  --nms_thresh 0.3
```

