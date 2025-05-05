import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageCopy, imageShow


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
        
        # Add Noise
        noise = np.random.normal(scale=opt.sigma/255, size=image.shape[:2])
        noisyImage = imageCopy(image)/255
        noisyImage[:,:,0] += noise
        noisyImage[:,:,1] += noise
        noisyImage[:,:,2] += noise
        noisyImage = (np.clip(noisyImage, 0, 1)*255).astype(np.uint8)

        # Create Low-Pass Filter (Kernel)
        kernelMean = np.array([[1,1,1], [1,1,1], [1,1,1]])/9
        kernelGaussian0 = np.array([[1,1,1], [1,2,1], [1,1,1]])/10
        kernelGaussian1 = np.array([[2,1,2], [1,4,1], [2,1,2]])/16
        kernelGaussian2 = np.array([[1,2,1], [2,4,2], [1,2,1]])/16
        
        # Apply Convolutional Operation
        imageMean = cv2.filter2D(noisyImage, ddepth=-1, kernel=kernelMean)
        imageGau0 = cv2.filter2D(noisyImage, ddepth=-1, kernel=kernelGaussian0)
        imageGau1 = cv2.filter2D(noisyImage, ddepth=-1, kernel=kernelGaussian1)
        imageGau2 = cv2.filter2D(noisyImage, ddepth=-1, kernel=kernelGaussian2)
        
        # Stack Image Horizontally
        imageStack = np.hstack([noisyImage, imageMean, imageGau0, imageGau1, imageGau2])
        
        # Show Result
        imageShow("Image Result", imageStack)
        

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    parser.add_argument("--sigma", type=int, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)