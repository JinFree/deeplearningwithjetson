import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageCopy, imageRead, imageShow


#########################################################################################


def drawRect(image, point1, point2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA) : # Line Type -> https://076923.github.io/posts/Python-opencv-18/
    result = imageCopy(image)
    return cv2.rectangle(result, point1, point2, color, thickness, lineType)


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

        # Determine Points
        point1 = (130, 197)
        point2 = (190, 259)

        # Draw Rectangles
        rect1 = drawRect(image, point1, point2, (0, 0, 255), 5) # 외쪽 상단과 오른쪽 하단 좌표를 이어주는 사각형 표현
        rect2 = drawRect(image, point1, point2, (0, 0, 255), 0) # 외쪽 상단과 오른쪽 하단 좌표를 이어주는 사각형 표현
        rect3 = drawRect(image, point1, point2, (0, 0, 255), -1) # 외쪽 상단과 오른쪽 하단 좌표를 이어주는 사각형 표현

        imageShow("Rectangle 1", rect1)
        imageShow("Rectangle 2", rect2)
        imageShow("Rectangle 3", rect3)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)