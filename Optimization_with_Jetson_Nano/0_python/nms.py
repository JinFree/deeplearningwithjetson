import timeit
import numpy as np

def nms_boxes(boxes, scores, score_threshold, nms_threshold, eta=1.0, top_k=0):
    """
    OpenCV cv2.dnn.NMSBoxes와 동일한 인터페이스의 순수 Python 구현.

    Args:
        boxes (List[List[float]]): [x, y, w, h] 형태의 박스 리스트
        scores (List[float]): 각 박스의 신뢰도 점수 리스트
        score_threshold (float): 점수 임계값 (이하 박스는 무시)
        nms_threshold (float): NMS에서 사용할 IoU 임계값
        eta (float): 어댑티브 임계값 계산용 (1.0이면 고정 임계값)
        top_k (int): 상위 k개 박스만 고려 (0이면 제한 없음)

    Returns:
        List[int]: 남겨진 박스의 인덱스 리스트
    """
    # 1) score_threshold보다 높은 박스 인덱스만 선택
    idxs = [i for i, s in enumerate(scores) if s > score_threshold]
    # 2) 점수 내림차순으로 정렬
    idxs.sort(key=lambda i: scores[i], reverse=True)
    # 3) top_k가 0보다 크면 상위 k개만 남김
    if top_k > 0:
        idxs = idxs[:top_k]

    def iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        # 교집합 영역 계산
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        wi = max(0.0, xi2 - xi1)
        hi = max(0.0, yi2 - yi1)
        inter = wi * hi
        # 합집합 영역 계산
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    picked = []
    adaptive_threshold = nms_threshold

    while idxs:
        # 4) 현재 가장 높은 점수 박스를 선택
        current = idxs.pop(0)
        picked.append(current)

        # 5) IoU가 adaptive_threshold 이하인 박스만 남김
        filtered = []
        for idx in idxs:
            if iou(boxes[current], boxes[idx]) <= adaptive_threshold:
                filtered.append(idx)
        idxs = filtered

        # 6) eta < 1.0인 경우 임계값을 업데이트
        if eta < 1.0:
            adaptive_threshold *= eta

    return picked


# 예제 박스와 점수
def generate_boxes_and_scores(num_boxes):
    import random
    random.seed(42)
    boxes = []
    scores = []
    for _ in range(num_boxes):
        x = random.random()
        y = random.random()
        w = random.random() * (1 - x)
        h = random.random() * (1 - y)
        boxes.append([round(x, 4), round(y, 4), round(w, 4), round(h, 4)])
        scores.append(round(random.random(), 4))
    return boxes, scores
boxes, scores = generate_boxes_and_scores(4096)

# NMS 실행
timer = timeit.Timer(nms_boxes(
    boxes, scores,
    score_threshold=0.5,
    nms_threshold=0.4,
    eta=1.0,
    top_k=0
).copy)
duration = timer.timeit(number=100)
print(f"100회 반복 평균: {duration/100:.6f}초")