import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageRead, imageShow, splitImage


#########################################################################################


def convertColor(image, flag=cv2.COLOR_BGR2GRAY) :
    return cv2.cvtColor(image, flag)


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

        # Split Each Channel
        b, g, r = splitImage(image)
        imageShow("B", b)
        imageShow("G", g)
        imageShow("R", r)

        # Convert to HSV
        imageHSV = convertColor(image, cv2.COLOR_BGR2HSV)
        imageShow("image.HSV", imageHSV)
        h, s, v = splitImage(imageHSV)
        imageShow("H", h)
        imageShow("S", s)
        imageShow("V", v)

        # Convert to HLS
        imageHLS = convertColor(image, cv2.COLOR_BGR2HLS)
        imageShow("Image-HLS", imageHLS)
        h2, l2, s2 = splitImage(imageHLS)
        imageShow("H2", h2)
        imageShow("L2", l2)
        imageShow("S2", s2)

        # Convert to Grayscale
        imageGray = convertColor(image, cv2.COLOR_BGR2GRAY)
        imageShow("imageGray", imageGray)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)