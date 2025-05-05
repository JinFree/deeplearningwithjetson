import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageCopy, imageRead, imageShow


#########################################################################################


def drawCircle(image, center, radius, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA) : # Line Type -> https://076923.github.io/posts/Python-opencv-18/
    result = imageCopy(image)
    return cv2.circle(result, center, radius, color, thickness, lineType)


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
        
        # Determine Center Point
        center1 = (160, 230)
        center2 = (160, 320)
        center3 = (160, 410)
        
        # Determin Radius
        radius = 32

        # Draw Circles
        circle1 = drawCircle(image, center1, radius, (0, 0, 255), 5) # 중심 좌표와 지름을 사용하여 원을 표현 
        circle2 = drawCircle(image, center2, radius, (0, 255, 255), 0) # 중심 좌표와 지름을 사용하여 원을 표현
        circle3 = drawCircle(image, center3, radius, (0, 255, 0), -1) # 중심 좌표와 지름을 사용하여 원을 표현

        imageShow("Circle-1", circle1)
        imageShow("Circle-2", circle2)
        imageShow("Circle-3", circle3)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)