import argparse

from os import listdir, makedirs
from os.path import join

from utils import *


#########################################################################################


def main(opt) :
    # Search for All Images
    imageNameList = listdir(opt.imagePath)
    
    # Create Directory for Saving Copied Image
    outputPath = join(opt.savePath, "Saved-Output", "Copied-Image")
    makedirs(outputPath, exist_ok=True)
    
    # Load Images
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)
        
        # Function-Calling
        processSingleImage(imagePath, join(outputPath, imageName))


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    parser.add_argument("--savePath", type=str, required=True, help="path for saving images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)