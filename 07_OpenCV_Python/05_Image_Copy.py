import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, setPixel


#########################################################################################


def imageCopy(src) :
    return np.copy(src)


#########################################################################################


def main(opt) :
    # Search for All images
    imageNameList = listdir(opt.imagePath)

    # Load Images
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)

        # Load Image
        image1 = imageRead(imagePath)
        image2 = image1
        image3 = imageCopy(image1)
        print(f"[Image '{imageName}' Loaded!]")
        
        # Get Memory Address
        print(f"Image 1 Memory Address : {id(image1)}")
        print(f"Image 2 Memory Address : {id(image2)}")
        print(f"Image 3 Memory Address : {id(image3)}")

        # Show Original & Copied Image
        imageShow("Image 1 before Function", image1)
        imageShow("Image 2 before Function", image2)
        imageShow("Image 3 before Function", image3)

        # Modify Pixel Values
        for i in range(image1.shape[1]) :
            for j in range(image1.shape[0]) :
                setPixel(image1, i, j, 255, 0)

        # Show Modified Image
        imageShow("Image 1 after Pixel Modification", image1)
        imageShow("Image 2 after Pixel Modification", image2)
        imageShow("Image 3 after Pixel Modification", image3)

        cv2.destroyAllWindows()

        # Load and Copy Image
        image1 = imageRead(imagePath)
        image2 = image1 # Call-by-Memory
        image3 = imageCopy(image1)

        # Show Image
        imageShow("Image 1 before Function", image1)
        imageShow("Image 2 before Function", image2)
        imageShow("Image 3 before Function", image3)

        # Convert Color
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        # Show Modified Image
        imageShow("Image 1 after Function", image1)
        imageShow("Image 2 after Function", image2)
        imageShow("Image 3 after Function", image3)
        
        # Get Image Shape
        print(f"Image 1 Image Shape : {image1.shape}")
        print(f"Image 2 Image Shape : {image2.shape}")
        print(f"Image 3 Image Shape : {image3.shape}")

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)