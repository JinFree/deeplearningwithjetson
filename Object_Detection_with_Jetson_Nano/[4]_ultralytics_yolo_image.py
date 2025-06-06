import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse

# =============================
# 1) 설정부
# =============================
# 클래스 이름 리스트.
CLASS_NAMES =  [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]



# =============================
# 2) TensorRT 엔진 로더 클래스
# =============================
class TrtUltralyticsYOLO:
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
        """엔진의 바인딩(입출력) 정보를 읽어 GPU 메모리, 호스트 메모리 버퍼를 할당."""
        self.bindings = []
        self.stream = cuda.Stream()
        self.host_inputs = []
        self.device_inputs = []
        self.host_outputs = []
        self.device_outputs = []

        for binding_idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding_idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            shape = tuple(self.engine.get_binding_shape(binding_idx))
            # 동적 배치(Dynamic shape)일 경우엔 컨텍스트에서 현재 배치에 맞게 set_binding_shape 호출 필요
            if self.engine.binding_is_input(binding_idx):
                # 입력은 일반적으로 (1, 3, H, W)
                size = trt.volume(shape) * np.dtype(dtype).itemsize
                # 호스트/디바이스 버퍼 생성
                host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype)
                device_mem = cuda.mem_alloc(size)
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                # 출력도 비슷하게 생성
                # -1(배치) 처리: 일반적으로 배치=1이므로 shape[0]=1로 가정. (동적 shape인 경우, inference 전에 set_binding_shape 필요)
                out_shape = tuple(shape)
                size = trt.volume(out_shape) * np.dtype(dtype).itemsize
                host_mem = cuda.pagelocked_empty(trt.volume(out_shape), dtype)
                device_mem = cuda.mem_alloc(size)
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

            self.bindings.append(int(device_mem))  # 디바이스 포인터(정수) 저장

        self.batch_size = 1

    def infer(self, input_image):
        """전처리된 단일 이미지를 GPU로 복사하여 TensorRT 추론, 결과를 반환."""
        # 1) 바인딩 셋팅 (입력 shape이 동적이라면)
        #    예: self.context.set_binding_shape(0, (1,3,INPUT_HEIGHT,INPUT_WIDTH))

        # 2) 호스트 입력 버퍼에 copy (평탄화)
        np.copyto(self.host_inputs[0], input_image.ravel())

        # 3) 호스트 → 디바이스 복사
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)

        # 4) inference 실행
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 5) 디바이스 → 호스트 복사
        for i, _ in enumerate(self.host_outputs):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.device_outputs[i], self.stream)

        # 6) 스트림 동기화
        self.stream.synchronize()

        # 7) 출력 버퍼를 NumPy 배열로 변환하여 반환 (list of 배열)
        output_arrays = []
        for host_mem in self.host_outputs:
            output_arrays.append(np.array(host_mem).reshape(self.engine.get_binding_shape(self.host_outputs.index(host_mem))))

        return output_arrays


# =============================
# 3) 전처리 & 후처리 함수
# =============================
def preprocess_image(cv_img, input_w, input_h):
    """
    OpenCV BGR 이미지를 받아 YOLOv11 입력 형태(1×3×input_h×input_w)로 변환.
    - BGR→RGB, 크기 조정, 정규화 (0~1), 채널 우선순서(C×H×W)로 변경
    """
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_w, input_h))
    img_normalized = img_resized.astype(np.float32) / 255.0
    # (H,W,C) → (C,H,W)
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    # 배치 차원 추가: (1,C,H,W)
    return np.expand_dims(img_transposed, axis=0)


def xywh2xyxy(box):
    """
    YOLO format (center_x, center_y, w, h) → (x1, y1, x2, y2)
    box: [cx, cy, w, h] (절대 픽셀 좌표 기준)
    """
    x_c, y_c, w, h = box
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return [x1, y1, x2, y2]


def postprocess(outputs, orig_w, orig_h, input_w, input_h, conf_thresh, nms_thresh, NUM_CLASSES=80):
    """
    TensorRT 출력 후처리
    - outputs: 리스트, [0]에는 (1, N, 5+NUM_CLASSES) 형태의 float32 출력. batch_size=1 가정
    - orig_w, orig_h: 원본 이미지 크기
    - input_w, input_h: 모델 입력 크기 (예: 416,416)
    """
    # 1) 출력 배열 가져오기
    preds = outputs[0].reshape(-1, 5 + NUM_CLASSES)  # (배치=1, num_preds, 5+클래스)
    # 2) objectness score와 class score 계산
    #    preds[:, 4] = objectness, preds[:,5:] = class_conf
    confs = preds[:, 4]
    class_probs = preds[:, 5:]  # (num_preds, NUM_CLASSES)
    class_ids = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(len(class_probs)), class_ids]
    # 3) 최종 confidence = objectness * class_score
    final_scores = confs * class_scores

    # 4) 임계값 이상의 인덱스만 선별
    mask = final_scores > conf_thresh
    filtered_preds = preds[mask]
    filtered_scores = final_scores[mask]
    filtered_class_ids = class_ids[mask]

    if len(filtered_preds) == 0:
        return []

    # 5) 좌표 변환: 모델 출력 (center_x, center_y, w, h)는 0~1로 정규화된 값이라 가정
    #    → 절대 픽셀 좌표: x*input_w → 원본비율로 매핑: x * (orig_w / input_w)
    boxes = []
    for i, det in enumerate(filtered_preds):
        cx = det[0] * orig_w / input_w
        cy = det[1] * orig_h / input_h
        bw = det[2] * orig_w / input_w
        bh = det[3] * orig_h / input_h
        x1, y1, x2, y2 = xywh2xyxy([cx, cy, bw, bh])
        boxes.append([x1, y1, x2 - x1, y2 - y1])  # NMS용: (x, y, w, h)

    # 6) OpenCV NMS 수행 (x, y, w, h → x1,y1,x2,y2 변환은 draw 시 처리)
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes,
        scores=filtered_scores.tolist(),
        score_threshold=conf_thresh,
        nms_threshold=nms_thresh
    )

    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            x2 = x + w
            y2 = y + h
            result.append({
                "box": [int(x), int(y), int(x2), int(y2)],
                "score": float(filtered_scores[i]),
                "class_id": int(filtered_class_ids[i])
            })
    return result


# =============================
# 4) 메인 함수: 추론 & 시각화
# =============================
def main(engine_path, image_path, input_width=416, input_height=416, conf_thresh=0.25, nms_thresh=0.45):
    # 1) 이미지 로드
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    orig_h, orig_w = orig_img.shape[:2]

    # 2) 전처리
    input_tensor = preprocess_image(orig_img, input_width, input_height)
    # --> shape = (1,3,INPUT_H,INPUT_W)

    # 3) TensorRT 엔진 로드 및 추론
    yolo = TrtUltralyticsYOLO(engine_path)
    outputs = yolo.infer(input_tensor)

    # 4) 후처리 (박스 리스트 반환)
    detections = postprocess(outputs, orig_w, orig_h, input_width, input_height, conf_thresh, nms_thresh)

    # 5) 결과 시각화
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class_id"]
        score = det["score"]
        color = (0, 255, 0)  # 초록색 박스
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(orig_img, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
        cv2.putText(orig_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # 6) 결과 출력
    cv2.imshow("YOLOv11 + TensorRT Inference", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv11 TensorRT + OpenCV Inference Script")
    parser.add_argument("--engine", type=str, default="yolo11n_fp16.engine", help="TensorRT 엔진 파일 경로")
    parser.add_argument("--image",  type=str, required=True, help="추론할 이미지 파일 경로")
    parser.add_argument("--input_width",  type=int, default=416, help="모델 입력 가로 크기")
    parser.add_argument("--conf_thresh",  type=float, default=0.25, help="객체 신뢰도 임계값")
    parser.add_argument("--nms_thresh",   type=float, default=0.45, help="NMS IoU 임계값")
    args = parser.parse_args()

    main(args.engine, args.image, args.input_width, args.input_width, args.conf_thresh, args.nms_thresh)
