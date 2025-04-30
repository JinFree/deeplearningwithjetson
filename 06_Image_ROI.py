import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, imageCopy


#########################################################################################


def CutRectROI(image, x1, y1, x2, y2) :
    return image[y1:y2, x1:x2]


def PasteRectROI(image, x1, y1, dst) :
    y2, x2 = image.shape[:2]
    dst[y1:y1+y2, x1:x1+x2] = image
    return dst


def makeBlackImage(image, color=False) :
    height, width = image.shape[0], image.shape[1]
    if color is True :
        return np.zeros((height, width, 3), np.uint8)
    else:
        if len(image.shape) == 2 :
            return np.zeros((height, width), np.uint8)
        else:
            return np.zeros((height, width, 3), np.uint8)
        

def fillPolyROI(image, points) :
    if len(image.shape) == 2 :
        channels = 1
    else:
        channels = image.shape[2]
    mask = makeBlackImage(image)
    ignoreMaskColor = (255,)*channels
    cv2.fillPoly(mask, points, ignoreMaskColor)
    return mask


def polyROI(image, points) :
    mask = fillPolyROI(image, points)
    return cv2.bitwise_and(image, mask)


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
        if image.shape[-1] != 3 :
            return ValueError(f"# Channel != {3}") # Filter RGBA Image
        imageShow("Image", image)

        # Show Region of Interest (ROI)
        roiX1 = 94 
        roiY1 = 274
        roiX2 = 224
        roiY2 = 366
        roiRect = CutRectROI(image, roiX1, roiY1, roiX2, roiY2)
        imageShow("ROI Rectangle", roiRect)

        # Paste ROI onto the Image
        image2 = imageCopy(image)
        roiNewX1 = 95
        roiNewY1 = 367
        image2 = PasteRectROI(roiRect, roiNewX1, roiNewY1, image2)
        imageShow("image2", image2)

        # Get Polygon ROI
        roiPoly1 = np.array([[(143,192), (126,212), (126,242), (143,261), 
                              (179,261), (192,242), (192,212), (179,192)]], dtype=np.int32)
        imageROIPoly1 = polyROI(image, roiPoly1)
        imageShow("Image ROI with Polygon 1", imageROIPoly1)

        # Get Polygon ROI
        point1 = (95, 367) 
        point2 = (225, 367)
        point3 = (225, 459)
        point4 = (95, 459)
        roiPoly2 = np.array([[point1, point2, point3, point4]], dtype=np.int32)
        imageROIPoly2 = polyROI(image, roiPoly2)
        imageShow("Image ROI with Polygon 2", imageROIPoly2)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)