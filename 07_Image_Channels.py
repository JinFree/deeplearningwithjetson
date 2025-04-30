import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageRead, imageShow


#########################################################################################


def splitImage(image) :
    return cv2.split(image)


def mergeImage(channel1, channel2, channel3) :
    return cv2.merge((channel1, channel2, channel3))


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
        imageShow("Image", image)

        # Split Channel
        b, g, r = splitImage(image)
        imageShow("Image-Blue", b) # Blue-Channel
        imageShow("Image-Green", g) # Green-Channel
        imageShow("Image-Red", r) # Red-Channel
        
        # Show Dimension of Each Channel
        print(f"Blue-Channel Image Dimension : {b.shape}")
        print(f"Green-Channel Image Dimension : {g.shape}")
        print(f"Red-Channel Image Dimension : {r.shape}")

        # Merge Channel
        image2 = mergeImage(b, g, r) # 2D Image -> 3D Image
        imageShow("Image-Final", image2)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)