import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, convertColor


#########################################################################################


def cannyEdge(image, threshold1=100, threshold2=200) :
    return cv2.Canny(image, threshold1, threshold2) # threshold1 -> 하단 임계값 / threshold2 -> 상단 임계값 (실질적으로 Edge를 판단하는 임계 값)


def autoCanny(image, sigma=0.33) :
    image = cv2.GaussianBlur(image, (3,3), 0) # ksize -> 3
    v = np.median(image) # Compute the Median of the Single channel Pixel Intensities
    lower = int(max(0, (1.0-sigma)*v)) # Apply Automatic Canny Edge Detection Using the Computed Median
    upper = int(min(255, (1.0+sigma)*v)) # Apply Automatic Canny Edge Detection Using the Computed Median
    edged = cv2.Canny(image, lower, upper) # Apply Automatic Canny Edge Detection Using the Computed Median
    return edged # Return the Edged Image


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
        
        # Convert into Grayscale Image
        image = convertColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny Filter
        imageCanny = cannyEdge(image)
        imageShow("Image Canny Edge Detection", imageCanny)
        
        # Apply Auto Canny Filter
        imageAutoCanny = autoCanny(image)
        imageShow("Image Auto Canny Edge Detection", imageAutoCanny)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)