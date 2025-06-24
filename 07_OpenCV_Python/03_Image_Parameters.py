import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageRead, imageShow


#########################################################################################


def imageParameters(imageName, image) :
    # Get Image Dimension (Grayscale or RGB-Scale)
    if len(image.shape) == 2 :
        height, width = image.shape
    elif len(image.shape) == 3 :
        height, width, channel = image.shape
    
    # Show Image Features
    print(f"{imageName}.shape is {image.shape}")
    print(f"{imageName}.shape[0] is Height: {height}")
    print(f"{imageName}.shape[1] is Width: {width}")
    
    # Show Image Dimension
    if len(image.shape) == 2 :
        print("This is Grayscale Image.")
        print(f"{imageName}.shape[2] is Not exist, So channel is 1")
    elif len(image.shape) == 3 :
        print("This is not Grayscale Image.")
        print(f"{imageName}.shape[2] is channels: {channel}")
    
    print(f"{imageName}.dtype is {image.dtype}")
    
    return height, width


#########################################################################################


def main(opt) :
    # Search for All images
    imageNameList = listdir(opt.imagePath)

    # Load Images
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)
        
        # RGB-Image
        image = imageRead(imagePath)
        height, width = imageParameters("Image", image)
        imageShow("Opened Image", image)
        print(f"< Height : {height} || Width : {width} >")
        
        print("#########################################################################################")
        
        # Grayscale-Image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = imageParameters("Image", image)
        imageShow("Opened Image", image)
        print(f"< Height : {height} || Width : {width} >")


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)