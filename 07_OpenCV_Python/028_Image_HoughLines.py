import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, imageCopy, convertColor, cannyEdge


#########################################################################################


def houghLines(image, rho=1, theta=np.pi/180, threshold=100) :
    return cv2.HoughLines(image, rho, theta, threshold) # img: 입력 이미지, 1 채널 바이너리 스케일 / rho: 거리 측정 해상도, 0~1 / theta: 각도, 라디안 단위 (np.pi/0~180) / 
                                                        # threshold: 직선으로 판단할 최소한의 동일 개수 (작은 값: 정확도 감소, 검출 개수 증가 / 큰 값: 정확도 증가, 검출 개수 감소)


def drawHoughLines(image, lines) :
    result = imageCopy(image)
    
    if len(image.shape) == 2 :
        result = convertColor(image, cv2.COLOR_GRAY2BGR)
    
    for i in range(len(lines)) : # Lines에서 Parameter 꺼내오기
        for rho, theta in lines[i] : # Parameter에 대한 좌표 변환 진행
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    return result


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

        # Get Edge-Filtered IMage
        imageEdge = cannyEdge(image, 100, 200)
        lines = houghLines(imageEdge, 1, np.pi/180, 125)
        
        # Get Image Lines (Hough-Lines)
        imageLines = drawHoughLines(image, lines)
        
        # Show Results
        imageShow("Image Edge", imageEdge)
        imageShow("Image Lines", imageLines)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)