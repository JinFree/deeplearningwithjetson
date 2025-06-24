import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, imageCopy, convertColor, cannyEdge, houghLines, drawHoughLines


#########################################################################################

# 허프 선 검출은 모든 점에 대해 수많은 선을 그어서 직선을 찾기 때문에 연산량이 무척 많음 
# 이를 개선하기 위한 방법이 확률적 허프 선 변환 -> cv2.HoughLinesP
def houghLinesP(image, rho=1.0, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=100) :
    return cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap) # img: 입력 이미지, 1 채널 바이너리 스케일 / rho: 거리 측정 해상도, 0~1 / theta: 각도, 라디안 단위 (np.pi/0~180) / 
                                                                                                             # threshold: 직선으로 판단할 최소한의 동일 개수 (작은 값: 정확도 감소, 검출 개수 증가 / 큰 값: 정확도 증가, 검출 개수 감소) /
                                                                                                             # minLineLength(optional): 선으로 인정할 최소 길이 / maxLineGap(optional): 선으로 판단할 최대 간격                                                                                            


def drawHoughLinesP(image, lines) :
    result = imageCopy(image)
    
    if len(image.shape) == 2 :
        result = convertColor(image, cv2.COLOR_GRAY2BGR)
    
    for i in range(len(lines)) : # Lines에서 Parameter 꺼내오기
        for x1, y1, x2, y2 in lines[i] :
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

        # Get Edge-Filtered Image
        imageEdge = cannyEdge(image, 100, 200)
        lines = houghLines(imageEdge, 1, np.pi/180, 50)
        linesP = houghLinesP(imageEdge, 1, np.pi/180, 50, 10, 50)
        
        # Get Image Lines (Hough-Lines)
        imageLines = drawHoughLines(image, lines)
        
        # Get Image Lines (Hough-Lines-Prob)
        imageLinesP = drawHoughLinesP(image, linesP)
        
        # Show Results
        imageShow("Image Edge", imageEdge)
        imageShow("Image Lines", imageLines)
        imageShow("Image Lines (Prob)", imageLinesP)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)