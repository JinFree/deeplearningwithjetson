import argparse

from os import listdir, makedirs
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, imageWrite, imageBilateralFilter, imageFiltering


#########################################################################################


def main(opt) :
    # Search for All Images
    imageNameList = listdir(opt.imagePath)
    
    # Create Directory for Saving Preprocessed Image
    outputPath = join(opt.savePath, "Preprocessed-Image")
    makedirs(outputPath, exist_ok=True)
    
    # Load Image
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)
        
        # Load Image
        image = imageRead(imagePath, cv2.IMREAD_COLOR)
        imageShow("Image", image)
        
        # Apply Bilateral Filter
        imageBF = imageBilateralFilter(image, 21, 75, 75)
        imageShow("Image BF", imageBF)
        
        # Increase Brigthness
        kernelBright = np.array([[0,0,0], [0,1,0], [0,0,0]])*1.125
        imageBright = imageFiltering(imageBF, kernelBright, -1)
        imageShow("Image Brightness", imageBright)
        
        # Apply Sharpening Filter
        kernelSharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        imageSharp = imageFiltering(imageBright, kernelSharp, -1)
        imageShow("Image Sharpen", imageSharp)
        
        # Save Preprocessed Image
        imageWrite(join(outputPath, imageName), imageSharp)


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True)
    parser.add_argument("--savePath", type=str, required=True)
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)