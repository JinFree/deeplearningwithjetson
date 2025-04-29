import argparse

from os import listdir
from os.path import join

import cv2

from utils import *


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
        
        # Convert BGR -> HLS
        imageHLS = convertColor(image, cv2.COLOR_BGR2HLS)
        imageShow("Image HLS", imageHLS)
        
        # imageHLS -> (H, L, S) 3-D Vector / Matrix
        H, L, S = splitImage(imageHLS)
        
        # (0,0) -> (200,200) @ L-Channels
        L[:200, :200] = 200
        
        # Concatenation / Aggregatopm / Merge
        imageHLS = mergeImage(H, L, S)
        imageShow("Image HLS Modified", imageHLS)
        
        cv2.destroyAllWindows()
       

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)