import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageRead, imageShow


#########################################################################################


def imageResize(image, dsize, interpolation=cv2.INTER_NEAREST) :
    return cv2.resize(image, dsize=dsize, interpolation=interpolation) # dsize: 절대 크기 (Pixel 기준) / interpolation: 보간법


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

        # Get Image Shape
        height, width = image.shape[:2]

        # Resize Image (Downsampling)
        imageNearest = imageResize(image, (height//2, width//2), cv2.INTER_NEAREST) # Nearest Interpolation
        imageBilinear = imageResize(image, (height//2, width//2), cv2.INTER_LINEAR) # Bilinear Interpolation
        imageBicubic = imageResize(image, (height//2, width//2), cv2.INTER_CUBIC) # Bicubic Interpolation
        
        # Show Image Size
        print(f"Original Image Size : {height}x{width}")
        print(f"Interpolated Image Size : {imageNearest.shape[0]}x{imageNearest.shape[1]}")

        # Show Results
        imageShow("Image Original", image)
        imageShow("Image Resized Nearest", imageNearest)
        imageShow("Image Resized Bilinear", imageBilinear)
        imageShow("Image Resized Bicubic", imageBicubic)
        
        # Resize Image (Upsampling)
        imageNearest = imageResize(imageNearest, (height*2, width*2), cv2.INTER_NEAREST) # Nearest Interpolation
        imageBilinear = imageResize(imageBilinear, (height*2, width*2), cv2.INTER_LINEAR) # Bilinear Interpolation
        imageBicubic = imageResize(imageBicubic, (height*2, width*2), cv2.INTER_CUBIC) # Bicubic Interpolation
        
        # Show Image Size
        print(f"Original Image Size : {height}x{width}")
        print(f"Interpolated Image Size : {imageNearest.shape[0]}x{imageNearest.shape[1]}")

        # Show Results
        imageShow("Image Original", image)
        imageShow("Image Resized Nearest", imageNearest)
        imageShow("Image Resized Bilinear", imageBilinear)
        imageShow("Image Resized Bicubic", imageBicubic)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)