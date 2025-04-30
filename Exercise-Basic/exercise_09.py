import argparse

from os import listdir
from os.path import join

import cv2

from utils import *


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
        
        if "01" in imageName :
            circles = houghCircles(backup, cv2.HOUGH_GRADIENT, dp=1, minDist=50, canny=50, threshold=50, minRadius=25, maxRadius=50)
        elif "02" in imageName :
            circles = houghCircles(backup, cv2.HOUGH_GRADIENT, dp=1, minDist=50, canny=45, threshold=25, minRadius=40, maxRadius=50)
            
        image = drawHoughCircles(backupBGR, circles)
        
        imageShow("Image", image)
        
        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)