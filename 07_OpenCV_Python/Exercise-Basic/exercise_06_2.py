import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageCopy, imageShow


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
        
        # Create Sharpening Kernel
        centerNine = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        centerFive = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        
        # Apply Convolutional Operation
        imageSharpNine = cv2.filter2D(image, -1, centerNine)
        imageSharpFive = cv2.filter2D(image, -1, centerFive)
        
        # Stack Image
        imageStack = np.hstack([image, imageSharpNine, imageSharpFive])
        
        # Show Result
        imageShow("Image Result", imageStack)
        

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)