import argparse

from os import listdir
from os.path import join

import cv2

from utils import *


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
        
        # Resize Image Using Bilinear Interpolation
        height, width = image.shape[:2]
        height, width = height//2, width//2
        newImage = imageResize(image, (width,height), interpolation=cv2.INTER_LINEAR)
        imageShow("Resized", newImage)
        
        # Convert BGR to Grayscale Image
        newImage = convertColor(newImage, cv2.COLOR_BGR2GRAY)
        imageShow("Grayscale", newImage)
        
        # Binarize Image
        newImage = imageThreshold(newImage, thresh=128, maxval=255, type=cv2.THRESH_BINARY)
        imageShow("Binarized", newImage)
        
        # Apply MorphologicalEx Opertion
        kernel = imageMorphologyKernel(cv2.MORPH_CROSS, size=5)
        newImage = imageMorphologyEx(newImage, op=cv2.MORPH_GRADIENT, kernel=kernel)
        imageShow("Morph Edge", newImage)
        
        # Warp Persepective Transformation
        srcPoint1 = [int(width*0.35), int(height*0.65)]
        srcPoint2 = [int(width*0.65), int(height*0.65)]
        srcPoint3 = [width, height]
        srcPoint4 = [0, height]
        dstPoint1 = [int(width*0.1), 0]
        dstPoint2 = [int(width*0.9), 0]
        dstPoint3 = [int(width*0.9), height]
        dstPoint4 = [int(width*0.1), height]
        
        # Aggregate Points
        srcPoints = np.float32([srcPoint1, srcPoint2, srcPoint3, srcPoint4])
        dstPoints = np.float32([dstPoint1, dstPoint2, dstPoint3, dstPoint4])
        
        # Apply Warp Perspective Transformation
        affineResult1 = imagePerspectiveTransformation(newImage, srcPoints, dstPoints)
        affineResult2 = imagePerspectiveTransformation(affineResult1, srcPoints, dstPoints, flags=cv2.WARP_INVERSE_MAP)
        affineResult3 = imagePerspectiveTransformation(affineResult1, dstPoints, srcPoints)
        
        # Show Results
        imageShow("Affine Result 1", affineResult1)
        imageShow("Affine Result 2", affineResult2)
        imageShow("Affine Result 3", affineResult3)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)