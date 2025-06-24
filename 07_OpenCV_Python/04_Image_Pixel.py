import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageRead, imageShow


#########################################################################################


def getPixel(image, x, y, c=None) :
    return image[y, x, c]


def setPixel(image, x, y, Value, c=None) :
    image[y, x, c] = Value
    return image


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
        if image.shape[-1] != 3 :
            return ValueError(f"# Channel != {3}") # Filter RGBA Image
        
        # Show Image
        imageShow("Opened Image", image)
        
        # Set Pixel Location (Index)
        x, y = 100, 200

        # Show Pixel Value
        pixelValue = getPixel(image, x, y)
        print(f"Pixel Value in {x}, {y} : {pixelValue}")

        # BGR-Image
        pixelValueBlue = getPixel(image, x, y, 0)
        pixelValueGreen = getPixel(image, x, y, 1)
        pixelValueRed = getPixel(image, x, y, 2)
        print(f"Pixel Value in {x}, {y} : Blue={pixelValueBlue}, Green={pixelValueGreen}, Red={pixelValueRed}")

        # Blue Channel
        for i in range(x, x+100) :
            for j in range(y, y+200) :
                image = setPixel(image, i, j, [(i-100)*1.2, 0, 0])
        imageShow("Image", image)

        # Green Channel
        for i in range(x, x+100) :
            for j in range(y, y+200) :
                image = setPixel(image, i, j, [0, (i-100)*1.2, 0])
        imageShow("Image", image)

        # Horizontal Line
        for i in range(0, image.shape[1]) :
            image = setPixel(image, i, y, 0, 0)
            image = setPixel(image, i, y, 255, 1)
            image = setPixel(image, i, y, 0, 2)
        imageShow("Image", image)

        # Vertical Line
        for j in range(0, image.shape[0]) :
            image = setPixel(image, x, j, [255, 0, 0])
        imageShow("Image", image)

        pixelValue = getPixel(image, x, y)
        print(f"Pixel Value in {x}, {y} : {pixelValue}")

        # BGR-Image
        pixelValueBlue = getPixel(image, x, y, 0)
        pixelValueGreen = getPixel(image, x, y, 1)
        pixelValueRed = getPixel(image, x, y, 2)
        print(f"Pixel Value in {x}, {y} : Blue={pixelValueBlue}, Green={pixelValueGreen}, Red={pixelValueRed}")

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)