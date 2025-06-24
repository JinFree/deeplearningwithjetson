import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, imageCopy, convertColor


#########################################################################################


def rangeColor(image, lower, upper) :
    result = imageCopy(image)
    return cv2.inRange(result, lower, upper) # Lower와 Upper-Bound를 설정하여 특정 생삭 영역을 추출 -> Binary Mask (사용 방법 -> https://deep-learning-study.tistory.com/123)


def splitColor(image, lower, upper) :
    result = imageCopy(image)
    mask = rangeColor(result, lower, upper)
    return cv2.bitwise_and(result, result, mask=mask) # Mask를 사용하여 서로 영역이 서로 겹치는 부분만 추출 (AND Operation -> 교집합)


#########################################################################################

def main(opt) :
    # Search for All images
    imageNameList = listdir(opt.imagePath)

    # Load Images
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)

        # Load Image
        image = imageRead(imagePath)
        imageShow("Image", image)

        # Set Range (Lower & Upper-Bound)
        lowerWhiteHSV = np.array([0, 0, 150]) # Lower-Bound
        upperWhiteHSV = np.array([179, 10, 255]) # Upper-Bound
        lowerYellowHSV = np.array([20, 50, 100]) # Lower-Bound
        upperYelloHSV = np.array([35, 255, 255]) # Upper-Bound

        # Convert Color (BGR -> HSV)
        imagHSV = convertColor(image, cv2.COLOR_BGR2HSV)

        # White와 Yellow Region 영역을 합치는 연산
        whiteHSVRegion = rangeColor(imagHSV, lowerWhiteHSV, upperWhiteHSV)
        yelloHSVRegion = rangeColor(imagHSV, lowerYellowHSV, upperYelloHSV)
        outputHSVRegion = whiteHSVRegion + yelloHSVRegion 

        # 원본 이미지에 영역을 표시
        whiteHSVOverlay = splitColor(imagHSV, lowerWhiteHSV, upperWhiteHSV)
        yellowHSVOverlay = splitColor(imagHSV, lowerYellowHSV, upperYelloHSV)
        outputHSVOverlay = whiteHSVOverlay + yellowHSVOverlay

        # Show Image
        imageShow("Output HSV Region", outputHSVRegion)
        imageShow("Output HSV Overlay", convertColor(outputHSVOverlay, cv2.COLOR_HSV2BGR))

        # Set Range (Lower & Upper-Bound)
        lowerWhiteHLS = np.array([0, 200, 0])
        upperWhiteHLS = np.array([179, 255, 255])
        lowerYellowHLS = np.array([15, 30, 115])
        upperYellowHLS = np.array([35, 204, 255])

        # Convert Color (BGR -> HLS)
        imageHLS = convertColor(image, cv2.COLOR_BGR2HLS)

        # White와 Yellow Region 영역을 합치는 연산
        whiteHLSRegion = rangeColor(imageHLS, lowerWhiteHLS, upperWhiteHLS)
        yellowHLSRegion = rangeColor(imageHLS, lowerYellowHLS, upperYellowHLS)
        outputHLSRegion = whiteHLSRegion + yellowHLSRegion

        # 원본 이미지에 영역을 표시
        whiteHLS_overlay = splitColor(imageHLS, lowerWhiteHLS, upperWhiteHLS)
        yellowHLS_overlay = splitColor(imageHLS, lowerYellowHLS, upperYellowHLS)
        outputHLS_overlay = whiteHLS_overlay + yellowHLS_overlay

        # Show Image
        imageShow("Output HLS Region", outputHLSRegion)
        imageShow("Output HLS Overlay", convertColor(outputHLS_overlay, cv2.COLOR_HLS2BGR))

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)