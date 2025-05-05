import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageCopy, imageRead, imageShow


#########################################################################################


def drawPolygon(image, points, isClosed, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA) : # Line Type -> https://076923.github.io/posts/Python-opencv-18/
    result = imageCopy(image)
    oldShape = points.shape
    points = points.reshape((-1, 1, 2))
    newShape = points.shape
    print(f"< Old Shape : {oldShape} || New Shape : {newShape} >")
    return cv2.polylines(result, [points], isClosed, color, thickness, lineType)


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

        # Determin Points of Polygon
        point1 = (143,192)
        point2 = (126,212)
        point3 = (126,242)
        point4 = (143,261)
        point5 = (179,261)
        point6 = (192,242)
        point7 = (192,212)
        point8 = (179,192)
        
        # Aggregate All Points
        points = np.vstack((point1, point2, point3, point4, point5, point6, point7, point8)).astype(np.int32) # 좌표의 집합을 통해 원하는 다각형을 구현
        pointsROI = np.array([[point1, point2, point3, point4, point5, point6, point7, point8]], dtype=np.int32) # 좌표의 집합을 통해 원하는 다각형을 구현

        # Draw Points
        poly1 = drawPolygon(image, points, False, (0, 0, 255), 5)
        poly2 = drawPolygon(image, points, True, (0, 0, 255), 5)
        poly3 = cv2.fillPoly(image, pointsROI, (0, 0, 255))

        imageShow("Poly-1", poly1)
        imageShow("Poly-2", poly2)
        imageShow("Poly-3", poly3)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)