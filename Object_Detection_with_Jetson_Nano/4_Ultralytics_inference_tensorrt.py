import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# =============================
# 1) 설정부
# =============================
# 클래스 이름 리스트.
def get_class_names():
    class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
    return class_names


# =============================
# 2) TensorRT 추론 클래스
# =============================
class TRTUltralyticsYOLO:
    def __init__(self, engine_path):    
        # TensorRT logger 생성
        self.logger = trt.Logger(trt.Logger.WARNING)
        # 엔진 로드
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            engine_data = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_data)
        # 컨텍스트 생성
        self.context = self.engine.create_execution_context()
        # I/O 바인딩 인덱스 및 크기 추출
        self._allocate_buffers()

    def _allocate_buffers(self):
        """
        엔진의 바인딩(입출력) 정보를 읽어
        - 입력/출력 바인딩 인덱스를 분리하여 저장
        - 호스트/디바이스 메모리 버퍼를 각각 할당
        """
        self.input_binding_idxs = []
        self.output_binding_idxs = []
        self.stream = cuda.Stream()

        self.host_inputs = []
        self.device_inputs = []
        self.host_outputs = []
        self.device_outputs = []

        for binding_idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            # binding shape (동적 배치인 경우, -1 포함)
            shape = tuple(self.engine.get_binding_shape(binding_idx))

            # 바인딩이 입력이면
            if self.engine.binding_is_input(binding_idx):
                # 입력: 일반적으로 (1, 3, INPUT_HEIGHT, INPUT_WIDTH)
                size = trt.volume(shape) * np.dtype(dtype).itemsize
                # 호스트/디바이스 버퍼 생성
                host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
                device_mem = cuda.mem_alloc(size)
                self.input_binding_idxs.append(binding_idx)
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)

            # 바인딩이 출력이면
            else:
                # 예시: (1, num_preds, 5 + NUM_CLASSES)
                # -1 (동적) 처리: TensorRT 엔진을 만들 때 고정해 두었다면, 여기에 -1이 없을 것
                size = trt.volume(shape) * np.dtype(dtype).itemsize
                host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
                device_mem = cuda.mem_alloc(size)
                self.output_binding_idxs.append(binding_idx)
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

            # 최종적으로 디바이스 포인터를 바인딩 리스트에 추가해야 execute_async_v2 때 사용 가능
            self.bindings = [int(dev) for dev in (self.device_inputs + self.device_outputs)]

        self.batch_size = 1

    def infer(self, input_image):
        """
        전처리된 단일 이미지를 GPU로 복사하여 TensorRT 추론을 수행하고,
        결과를 NumPy 배열 리스트로 반환합니다.
        """
        # (1) 입력 바인딩이 동적(shape에 -1이 있을 경우)이라면
        #     ex) if self.engine.is_shape_binding(binding_idx): 
        #         self.context.set_binding_shape(binding_idx, (1,3,INPUT_HEIGHT,INPUT_WIDTH))

        # (2) 호스트 입력 버퍼에 전처리된 이미지(flatten) 복사
        np.copyto(self.host_inputs[0], input_image.ravel())

        # (3) 호스트 → 디바이스 메모리 복사
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)

        # (4) TensorRT 추론 실행
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # (5) 디바이스 → 호스트 메모리 복사
        for i, output_dev in enumerate(self.device_outputs):
            cuda.memcpy_dtoh_async(self.host_outputs[i], output_dev, self.stream)

        # (6) 스트림 동기화
        self.stream.synchronize()

        # (7) 출력 버퍼들을 NumPy 배열로 reshape
        output_arrays = []
        for i, host_mem in enumerate(self.host_outputs):
            binding_idx = self.output_binding_idxs[i]
            # 컨텍스트에 동적 shape인 경우가 있다면, 실제 바인딩 크기를 get
            out_shape = self.context.get_binding_shape(binding_idx)
            # 예: out_shape = (1, num_preds, 5+NUM_CLASSES)
            output_arrays.append(
                np.array(host_mem).reshape(out_shape)
            )
        return output_arrays


# =============================
# 3) 전처리 & 후처리 함수
# =============================
def preprocess_image(cv_img, input_w, input_h):
    """
    OpenCV BGR 이미지를 받아 YOLO 입력 형태(1 x 3 x input_h x input_w)로 변환.
    - BGR→RGB, 크기 조정, 정규화 (0~1), 채널 우선순서(C x H x W)로 변경
    """
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_w, input_h))
    img_normalized = img_resized.astype(np.float32) / 255.0
    # (H,W,C) → (C,H,W)
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    # 배치 차원 추가: (1,C,H,W)
    return np.expand_dims(img_transposed, axis=0)


def postprocess(outputs, conf_thresh, nms_thresh):
    results = outputs[0].transpose()  # (N, C) -> (C, N)
    if len(results[0]) != 5:
        class_filtered_results = []
        for detection in results:
            class_id = detection[4:].argmax()
            confidence_score = detection[4:].max()
            new_detection = np.append(detection[:4],[class_id,confidence_score])
            class_filtered_results.append(new_detection)
        results = np.array(class_filtered_results)

    boxes = results[:, :4]  # [x1, y1, x2, y2]
    conf_scores = results[:, -1]  # confidence scores
    # 겹치는 박스 제거를 위한 NMS와 thresholding
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf_scores.tolist(), score_threshold=conf_thresh, nms_threshold=nms_thresh)

    if len(indices) == 0:
        return np.empty((0, 6))  # No detections after NMS
    return results[indices.flatten()]


def visualize(class_names, results, orig_image, orig_h, orig_w, input_width=416, input_height=416):
    result_image = orig_image.copy()
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/input_width * orig_w
    cy = cy/input_height * orig_h
    w = w/input_width * orig_w
    h = h/input_height * orig_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    for box, class_idx, score in zip(zip(x1, y1, x2, y2), class_id, confidence):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_idx = int(class_idx)
        score = "{:.2f}".format(score)
        cv2.rectangle(result_image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0, 0),1)
        cv2.putText(result_image,class_names[class_idx]+' '+score,(x1,y1-17),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),1)
    return result_image


# =============================
# 4) 메인 함수: 추론 & 시각화
# =============================

def main(model_path, image_path, input_width, input_height, object_threshold, iou_threshold):
    # 1) 이미지 로드
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    orig_h, orig_w = orig_img.shape[:2]

    # 2) 전처리
    start = time.perf_counter() 
    input_tensor = preprocess_image(orig_img, input_width, input_height)
    end = time.perf_counter() 
    print(f"Preprocessing time: {end - start:.4f} seconds")
    
    # 3) ONNX 모델 로드 및 추론
    yolo = TRTUltralyticsYOLO(model_path)
    outputs = yolo.infer(input_tensor)
    outputs = yolo.infer(input_tensor)
    outputs = yolo.infer(input_tensor)
    start = time.perf_counter() 
    outputs = yolo.infer(input_tensor)
    end = time.perf_counter() 
    print(f"Inference time: {end - start:.4f} seconds")
    
    # 4) 후처리 (박스 리스트 반환)
    start = time.perf_counter() 
    detections = postprocess(outputs, object_threshold, iou_threshold)
    end = time.perf_counter() 
    print(f"Postprocessing time: {end - start:.4f} seconds")
    
    if detections.size == 0:
        print("No objects detected.")
        return
    class_names = get_class_names()
    result_image = visualize(class_names, detections, orig_img, orig_h, orig_w)
    cv2.imwrite('result.jpg', result_image)
    
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultralytics YOLO TensorRT + OpenCV Inference Script")
    parser.add_argument("--model", type=str, required=True, help="engine 파일 경로")
    parser.add_argument("--image",  type=str, required=True, help="추론할 이미지 파일 경로")
    parser.add_argument("--input_width",  type=int, default=416, help="모델 입력 가로 크기")
    parser.add_argument("--conf_thresh",  type=float, default=0.25, help="객체 신뢰도 임계값")
    parser.add_argument("--nms_thresh",   type=float, default=0.45, help="NMS IoU 임계값")
    args = parser.parse_args()

    main(args.model, args.image, args.input_width, args.input_width, args.conf_thresh, args.nms_thresh)
