import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, convertColor, rangeColor, splitColor


#########################################################################################


def main(opt) :
    # Search for All Images
    imageNameList = listdir(opt.imagePath)
    
    # Load Images
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)

        # Load Image
        image = imageRead(imagePath)
        imageShow("Image", image)
        
        # Convert BGR to HLS
        imageHLS = convertColor(image, cv2.COLOR_BGR2HLS)
        
        # Set Range (Lower & Upper-Bound)
        lowerWhiteHLS = np.array([0, 20, 20])
        upperWhiteHLS = np.array([15, 255, 255])
        lowerRedHLS = np.array([165, 20, 20])
        upperRedHLS = np.array([179, 255, 255])
        
        # Generate Mask
        whiteHLSRegion = rangeColor(imageHLS, lowerWhiteHLS, upperWhiteHLS)
        redHLSRegion = rangeColor(imageHLS, lowerRedHLS, upperRedHLS)
        
        # 영역 표시
        whiteHLSOverlay = splitColor(imageHLS, lowerWhiteHLS, upperWhiteHLS)
        redHLSOverlay = splitColor(imageHLS, lowerRedHLS, upperRedHLS)
        
        # Sum Generated Mask Region
        outputHLSRegion = whiteHLSRegion + redHLSRegion
        outputHLSOverlay = whiteHLSOverlay + redHLSOverlay
        
        # Show Results
        imageShow("Output Red Light Region", outputHLSRegion)
        imageShow("Output Red Light Overlay", outputHLSOverlay)


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)