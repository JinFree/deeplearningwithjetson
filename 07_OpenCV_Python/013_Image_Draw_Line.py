import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageCopy, imageRead, imageShow


#########################################################################################


def drawLine(image, point1, point2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA) : # Line Type -> https://076923.github.io/posts/Python-opencv-18/
    result = imageCopy(image)
    return cv2.line(result, point1, point2, color, thickness, lineType)


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

        # Draw Lines
        point1 = (130, 197)
        point2 = (130, 259)
        point3 = (190, 259)
        point4 = (190, 197)

        # 좌포를 지정하여 Point 1 -> Point 2를 연결하는 선을 표현
        line = drawLine(image, point1, point2, (0, 0, 255), 5)
        imageShow("Line", line)
        
        # 좌포를 지정하여 Point 2 -> Point 3를 연결하는 선을 표현
        line = drawLine(line, point2, point3, (0, 0, 255), 5)
        imageShow("Line", line)
        
        # 좌포를 지정하여 Point 3 -> Point 4를 연결하는 선을 표현
        line = drawLine(line, point3, point4, (0, 0, 255), 5)
        imageShow("Line", line)
        
        # 좌포를 지정하여 Point 4 -> Point 1을 연결하는 선을 표현
        line = drawLine(line, point4, point1, (0, 0, 255), 5)
        imageShow("Line", line)
        
        cv2.destroyAllWindows()


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)