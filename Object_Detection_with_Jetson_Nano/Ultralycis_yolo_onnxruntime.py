import cv2
import onnxruntime as ort
import numpy as np

# 1) 클래스 이름 로드
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

# 2) ONNX Runtime 세션 생성
def get_session(model_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider']):
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name  # 보통 'images'
    return session, input_name


# 3) 이미지 로드 및 전처리 (416x416 리사이즈, RGB, 정규화)
def preprocessing(image_path, input_width=416, input_height=416):
    orig_image = cv2.imread(image_path)
    orig_h, orig_w = orig_image.shape[:2]
    img = cv2.resize(orig_image, (input_width, input_height))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = np.transpose(img_rgb, (2, 0, 1))[None, ...]  # (1,3,416,416)
    return orig_image, orig_h, orig_w, input_tensor

# 4) 추론 실행
def inference_image(session, input_name, input_tensor):
    outputs = session.run(None, {input_name: input_tensor})
    return outputs[0] # outputs[0].shape == (1, C, N)  e.g. (1,24,8400)

# 5) 결과 후처리
def get_detection_output(outputs, object_threshold, iou_threshold):
    results = outputs.transpose()  # (N, C) -> (C, N)
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
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), conf_scores.tolist(), score_threshold=object_threshold, nms_threshold=iou_threshold)

    if len(indices) == 0:
        return np.empty((0, 6))  # No detections after NMS
    return results[indices.flatten()]

# 6) 결과 시각화
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

def main(model_path, image_path, input_width, input_height, object_threshold, iou_threshold):
    class_names = get_class_names()
    session, input_name = get_session(model_path)
    orig_image, orig_h, orig_w, input_tensor = preprocessing(image_path, input_width, input_height)
    outputs = inference_image(session, input_name, input_tensor)
    results = get_detection_output(outputs, object_threshold, iou_threshold)
    if results.size == 0:
        print("No objects detected.")
        return
    result_image = visualize(class_names, results, orig_image, orig_h, orig_w)
    cv2.imwrite('result.jpg', result_image)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv11 ONNX + OpenCV Inference Script")
    parser.add_argument("--onnx", type=str, default="yolo11n_fp16.engine", help="ONNX 파일 경로")
    parser.add_argument("--image",  type=str, required=True, help="추론할 이미지 파일 경로")
    parser.add_argument("--input_width",  type=int, default=416, help="모델 입력 가로 크기")
    parser.add_argument("--conf_thresh",  type=float, default=0.25, help="객체 신뢰도 임계값")
    parser.add_argument("--nms_thresh",   type=float, default=0.45, help="NMS IoU 임계값")
    args = parser.parse_args()

    main(args.onnx, args.image, args.input_width, args.input_width, args.conf_thresh, args.nms_thresh)
