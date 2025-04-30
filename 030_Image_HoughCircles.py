import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageRead, imageCopy, convertColor


#########################################################################################


def houghCircles(image, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, canny=50, threshold=30, minRadius=0, maxRadius=0) :
    circles = cv2.HoughCircles(image, method, dp, minDist, param1=canny, param2=threshold, minRadius=minRadius, maxRadius=maxRadius) 
    return circles # img: 입력 이미지, 1채널 배열 / method: 검출 방식 선택 (현재 cv2.HOUGH_GRADIENT만 가능) /
                   # dp: 입력 영상과 경사 누적의 해상도 반비례율, 1: 입력과 동일, 값이 커질수록 부정확 /
                   # minDist: 원들 중심 간의 최소 거리 (0: 에러, 0이면 동심원이 검출 불가하므로) /
                   # circles(optional): 검출 원 결과, N x 1 x 3 부동 소수점 배열 (x, y, 반지름) /
                   # param1(optional): 캐니 엣지에 전달할 스레시홀드 최대 값 (최소 값은 최대 값의 2배 작은 값을 전달) /
                   # param2(optional): 경사도 누적 경계 값 (값이 작을수록 잘못된 원 검출) /
                   # minRadius / maxRadius(optional): 원의 최소 반지름 / 최대 반지름 (0이면 이미지 전체의 크기)


def drawHoughCircles(image, circles) :
    result = imageCopy(image)
    
    if circles is None :
        return result
    
    for i in circles[0,:] : # Circle에서부터 Parameter 가져오기
        cx = int(i[0])
        cy = int(i[1])
        rr = int(i[2])
        cv2.circle(result, (cx,cy), rr, (0, 255, 0), 2)
        cv2.circle(result, (cx,cy), 2, (0, 0, 255), -1)
    
    return result


def nothing(x) :
    pass


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
        backupBGR = imageCopy(image)
        backup = convertColor(backupBGR, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("Image", cv2.WINDOW_GUI_EXPANDED)

        # Set Track Bar
        cv2.createTrackbar("minDist", "Image", 50, 100, nothing)
        cv2.createTrackbar("canny", "Image", 50, 100, nothing)
        cv2.createTrackbar("threshold", "Image", 50, 100, nothing)
        cv2.createTrackbar("minRadius", "Image", 50, 450, nothing)
        cv2.createTrackbar("maxRadius", "Image", 50, 450, nothing)

        switch = "0:OFF\n1:On"
        cv2.createTrackbar(switch, "Image", 1, 1, nothing)

        while True :
            cv2.imshow("Image", image)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            minDist = cv2.getTrackbarPos("minDist", "Image")
            canny = cv2.getTrackbarPos("canny", "Image")
            threshold = cv2.getTrackbarPos("threshold", "Image")
            minRadius = cv2.getTrackbarPos("minRadius", "Image")
            maxRadius = cv2.getTrackbarPos("maxRadius", "Image")
            s = cv2.getTrackbarPos(switch, "Image")

            if s == 1 :
                circles = houghCircles(backup, cv2.HOUGH_GRADIENT, 1, minDist+1, canny+1, threshold+1, minRadius, maxRadius)
                image = drawHoughCircles(backupBGR, circles)
            else :
                image = backupBGR

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)