import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageRead, imageShow, drawCircle, drawText


#########################################################################################.


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
        
        # (195, 255)와 (280, 255) 지점에 반지름이 10인 파란색 원을 덧씌워 그리세요
        image = drawCircle(image, (195, 255), 10, (255, 0, 0), -1)
        imageShow("Image Circle-1", image)
        
        image = drawCircle(image, (280, 255), 10, (255, 0, 0), -1)
        imageShow("Image Circle-2", image)
        
        # (125, 200) 지점에 두께는 3인 검정색 문자열 “OpenCV Exercise”를 입력하세요.
        image = drawText(image, "OpenCV Exericise", (125, 200))
        imageShow("Image Text", image)
        
        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)