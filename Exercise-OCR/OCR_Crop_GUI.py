import argparse

from os import listdir, makedirs
from os.path import join

import cv2

from utils import *


#########################################################################################


def main(opt) :
    # Search for All Images
    imageNameList = listdir(opt.imagePath)
    
    # Create Directory for Saving Copied Image
    outputPath = join(opt.savePath, "ROI")
    makedirs(outputPath, exist_ok=True)
    
    # Load Images
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)
        
        # Load Image
        image = imageRead(imagePath)
        
        # Get ROI Features
        x, y, w, h = cv2.selectROI("Image", image, False)
        print(f"< ROI X Coord. : {x} | ROI Y Coord. : {y} | ROI Width : {w} | ROI Height : {h} >")
        
        # Crop ROI
        roi = image[y:y+h, x:x+w]
        
        # Show ROI Image
        cv2.imshow("Cropped", roi)
        cv2.imwrite(join(outputPath, imageName.replace(".", "-roi.")), roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    parser.add_argument("--savePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)