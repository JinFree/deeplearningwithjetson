import cv2
import numpy as np

from utils import setPixel, imageShow


#########################################################################################


def imageThreshold(image, thresh=128, maxval=255, type=cv2.THRESH_BINARY) : # thres -> 임계값 / maxval -> 임계값을 넘었을 때 적용할 값
    _, res = cv2.threshold(image, thresh=thresh, maxval=maxval, type=type)
    return res


#########################################################################################


def main() :
    # Create Image
    image = np.zeros((512, 512), np.uint8)
    
    for i in range(0, 512) :
        for j in range(0, 256) :
            image = setPixel(image, i, j, j)
        for j in range(256, 512) :
            image = setPixel(image, i, j, j-256)

    # Show Generated Image
    imageShow("Image", image)

    # Apply Different Threshold Method
    imageThreshBinary = imageThreshold(image, 128, 255, cv2.THRESH_BINARY) # 이미지 이진화
    imageShow("Image THRESH_BINARY", imageThreshBinary)

    imageThreshBinaryInv = imageThreshold(image, 128, 255, cv2.THRESH_BINARY_INV) # 이미지 이진화 (반전된 결과)
    imageShow("Image THRESH_BINARY_INV", imageThreshBinaryInv)

    imageThresTrunc = imageThreshold(image, 128, 255, cv2.THRESH_TRUNC) # 임계값 보다 크면 임계값으로 지정하고, 낮으면 배경영상을 그대로 이용
    imageShow("Image THRESH_TRUNC", imageThresTrunc)

    imageThreshToZero = imageThreshold(image, 128, 255, cv2.THRESH_TOZERO) # 임계값보다 크면 배경영상, 낮으면 검은색을 출력
    imageShow("Image THRESH_TOZERO", imageThreshToZero)

    imageThreshToZeroInv = imageThreshold(image, 128, 255, cv2.THRESH_TOZERO_INV) # 임계값보다 작으면 배경영상 크면 검은색
    imageShow("Image THRESH_TOZERO_INV", imageThreshToZeroInv)

    cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Excute Main Function
    main()