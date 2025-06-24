import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, imageEdgePrewitt, imageEdgeSobel, imageEdgeScharr, imageEdgeLaplacianCV, imageThreshold


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
        
        # Apply Edge Filter
        edgePrewitt = imageEdgePrewitt(image)
        edgeSobel = imageEdgeSobel(image)
        edgeScharr = imageEdgeScharr(image)
        edgeLap = imageEdgeLaplacianCV(image)
        
        # Threshold
        edgePrewitt = imageThreshold(edgePrewitt)
        edgeSobel = imageThreshold(edgeSobel)
        edgeScharr = imageThreshold(edgeScharr)
        edgeLap = imageThreshold(edgeLap)
        
        # Min-Max Norm [0,1]
        maskPrewitt = edgePrewitt/255
        maskSobel = edgeSobel/255
        maskScharr = edgeScharr/255
        maskLap = edgeLap/255
        
        # output = input*(1-mask) + edge
        imagePrewitt = np.clip(image*(1-maskPrewitt) + edgePrewitt, 0, 255).astype(np.uint8)
        imageSobel = np.clip(image*(1-maskSobel) + edgeSobel, 0, 255).astype(np.uint8)
        imageScharr = np.clip(image*(1-maskScharr) + edgeScharr, 0, 255).astype(np.uint8)
        imageLap = np.clip(image*(1-maskLap) + edgeLap, 0, 255).astype(np.uint8)
        
        # Show Results
        imageShow("Prewitt", imagePrewitt)
        imageShow("Sobel", imageSobel)
        imageShow("Scharr", imageScharr)
        imageShow("Laplacian", imageLap)
        
        

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)