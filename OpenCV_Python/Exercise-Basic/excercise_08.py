import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import *


#########################################################################################


def main(opt) :
    # Search for All Images
    imageNameList = listdir(opt.imagePath)
    
    # Load Image
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)
        
        # Load Image
        image = imageRead(imagePath, cv2.IMREAD_GRAYSCALE) # 2D Array
        
        # Add Noise
        noise = np.random.normal(scale=opt.sigma/255, size=image.shape[:2])
        noisyImage = imageCopy(image)/255
        noisyImage[:,:] += noise
        noisyImage = (np.clip(noisyImage, 0, 1)*255).astype(np.uint8)
        
        # Binarize Image
        noisyImageBin = imageThreshold(noisyImage)
        
        # 3x3 Rect Kernel
        kernel = np.ones((3,3), dtype=np.uint8)
        
        # Apply Morphological Operation
        noisyImageDil = imageDilation(noisyImageBin, kernel, 1)
        noisyImageEro = imageErosion(noisyImageBin, kernel, 1)
        noisyImageOpen = imageOpening(noisyImageBin, 1)
        noisyImageClose = imageClosing(noisyImageBin, 1)
        noisyImageGrad = imageMorphologicalGradient(noisyImageBin)
        
        # Show Results
        imageShow("Image Threshold", noisyImageBin)
        imageShow("Image Dilation", noisyImageDil)
        imageShow("Image Erosion", noisyImageEro)
        imageShow("Image Opening", noisyImageOpen)
        imageShow("Image Closing", noisyImageClose)
        imageShow("Image Gradient", noisyImageGrad)
        
        cv2.destroyAllWindows()
        
        # Get Kernels for Morphological Operation
        kernelRect = imageMorphologyKernel(cv2.MORPH_RECT, 5)
        kernelEllipse = imageMorphologyKernel(cv2.MORPH_ELLIPSE, 5)
        kernelCross = imageMorphologyKernel(cv2.MORPH_CROSS, 5)
        
        # Apply Opening Morphological Operation
        imageOpenRect = imageMorphologyEx(noisyImageBin, cv2.MORPH_OPEN, kernelRect) # 열림 연산
        imageOpenEllipse = imageMorphologyEx(noisyImageBin, cv2.MORPH_OPEN, kernelEllipse) # 열림 연산
        imageOpenCross = imageMorphologyEx(noisyImageBin, cv2.MORPH_OPEN, kernelCross) # 열림 연산

        # Show Results
        imageShow("Image Threshold", noisyImageBin)
        imageShow("MORPH_RECT Kernel", kernelRect)
        imageShow("Image Opening (Rect)", imageOpenRect)
        imageShow("MORPH_ELLIPSE Kernel", kernelEllipse)
        imageShow("Image Opening (Ellipse)", imageOpenEllipse)
        imageShow("MORPH_CROSS Kernel", kernelCross)
        imageShow("Image Opening (Cross)", imageOpenCross)

        cv2.destroyAllWindows()
        
        # Apply Closing Morphological Operation
        imageCloseRect = imageMorphologyEx(noisyImageBin, cv2.MORPH_CLOSE , kernelRect) # 닫힘 연산
        imageCloseEllipse = imageMorphologyEx(noisyImageBin, cv2.MORPH_CLOSE, kernelEllipse) # 닫힘 연산 
        imageCloseCross = imageMorphologyEx(noisyImageBin, cv2.MORPH_CLOSE, kernelCross) # 닫힘 연산

        # Show Results
        imageShow("Image Threshold", noisyImageBin)
        imageShow("MORPH_RECT Kernel", kernelRect)
        imageShow("Image Closing (Rect)", imageCloseRect)
        imageShow("MORPH_ELLIPSE Kernel", kernelEllipse)
        imageShow("image Closing (Ellipse)", imageCloseEllipse)
        imageShow("MORPH_CROSS Kernel", kernelCross)
        imageShow("image Closing (Cross)", imageCloseCross)

        cv2.destroyAllWindows()
        
        # Apply Gradient Morphological Operation
        imageGradientRect = imageMorphologyEx(noisyImageBin, cv2.MORPH_GRADIENT , kernelRect) # 그레디언트 연산
        imageGradientEllipse = imageMorphologyEx(noisyImageBin, cv2.MORPH_GRADIENT , kernelEllipse) # 그레디언트 연산
        imageGradientCross = imageMorphologyEx(noisyImageBin, cv2.MORPH_GRADIENT , kernelCross) # 그레디언트 연산

        imageShow("Image Threshold", noisyImageBin)
        imageShow("MORPH_RECT Kernel", kernelRect)
        imageShow("Image Gradient (Rect)", imageGradientRect)
        imageShow("MORPH_ELLIPSE Kernel", kernelEllipse)
        imageShow("Image Gradient (Ellipse)", imageGradientEllipse)
        imageShow("MORPH_CROSS Kernel", kernelCross)
        imageShow("Image Gradient (Cross)", imageGradientCross)

        cv2.destroyAllWindows()
        

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True)
    parser.add_argument("--sigma", type=int, default=15)
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)