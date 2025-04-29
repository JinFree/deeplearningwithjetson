import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, imageCopy, convertColor, cannyEdge, polyROI, houghLinesP


#########################################################################################


def splitTwoSideLines(lines, slopeThreshold=(5*np.pi/180)) :
    lefts, rights = [], []
    for line in lines :
        x1 = line[0,0]
        y1 = line[0,1]
        x2 = line[0,2]
        y2 = line[0,3]
        if (x2-x1) == 0 :
            continue
        slope = (float)(y2-y1)/(float)(x2-x1)
        if abs(slope) < slopeThreshold :
            continue
        if slope <= 0:
            lefts.append([slope, x1, y1, x2, y2])
        else:
            rights.append([slope, x1, y1, x2, y2])
    return lefts, rights


def medianPoint(x) :
    if len(x) == 0 :
        return None
    else:
        xx = sorted(x)
        return xx[(int)(len(xx)/2)]


def interpolate(x1, y1, x2, y2, y) :
    return int((y-y1)*(x2-x1)/(y2-y1) + x1)


def lineFitting(image, lines, color=(0,0,255), thickness=3, slopeThreshold=(5*np.pi/180)) :
    result = imageCopy(image)
    height = image.shape[0]
    lefts, rights = splitTwoSideLines(lines, slopeThreshold)
    left, right = medianPoint(lefts), medianPoint(rights)
    minY, maxY = int(height*0.6), height
    if left is not None:
        minXLeft = interpolate(left[1], left[2], left[3], left[4], minY)
        maxXLeft = interpolate(left[1], left[2], left[3], left[4], maxY)
        cv2.line(result, (minXLeft, minY), (maxXLeft, maxY), color, thickness)
    if right is not None:
        minXRight = interpolate(right[1], right[2], right[3], right[4], minY)
        maxXRight = interpolate(right[1], right[2], right[3], right[4], maxY)
        cv2.line(result, (minXRight, minY), (maxXRight, maxY), color, thickness)
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
        
        # Show Loaded Image
        imageShow("Image", image)

        # Convert BGR to Grayscale Image
        imageGray = convertColor(image, cv2.COLOR_BGR2GRAY)

        # Get Edge Map
        imageEdge = cannyEdge(imageGray, 100, 200)

        # Set Points
        height, width = image.shape[:2]
        point1 = (width*0.45, height*0.65)
        point2 = (width*0.55, height*0.65)
        point3 = (width, height*1.0)
        point4 = (0, height*1.0)

        # Get ROI Area
        roiCorners = np.array([[point1, point2, point3, point4]], dtype=np.int32)
        imageROI = polyROI(imageEdge, roiCorners)

        # Get Lines
        lines = houghLinesP(imageROI, 1, np.pi/180, 40)

        # Draw Lane
        imageLane = lineFitting(image, lines, (0, 0, 255), 5, 30*np.pi/180)

        # Show Result
        imageShow("imageLane", imageLane)


        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)