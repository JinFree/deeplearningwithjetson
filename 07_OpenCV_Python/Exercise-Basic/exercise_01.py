import argparse

from os import listdir
from os.path import join

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

        # BGR-Space Image (3-D Matrix)
        # (0,0) -> (200,200) => (0,0,0)
        
        # For Loop
        for i in range(200) :
            for j in range(200) :
                for k in range(3) :
                    image[i, j, k] = 0
                    
        imageShow("Image (For-Loop)", image)

        # Load Image
        image = imageRead(imagePath)
        
        # Pixel Modification Using Slicing
        image[0:200, 0:200, :] = 0
        imageShow("Image (Slicing)", image)


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)