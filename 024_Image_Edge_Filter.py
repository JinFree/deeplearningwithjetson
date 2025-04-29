import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, convertColor


#########################################################################################


def imageFiltering(image, kernel, ddepth=-1) :
    return cv2.filter2D(image, ddepth, kernel)


def imageEdgePrewitt(image) :
    kernelX = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]], np.float32) # x 방향으로의 Kernel 생성
    kernelY = np.array([[-1, -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]], np.float32) # y 방향으로의 Kernel 생성
    imageDeltaX = imageFiltering(image, kernelX) # x 방향으로의 Edge Filter
    imageDeltaY = imageFiltering(image, kernelY) # y 방향으로의 Edge Filter
    return imageDeltaX + imageDeltaY


def imageEdgeSobel(image) :
    imageDeltaX = cv2.Sobel(image, -1, 1, 0, ksize=3) # x 방향으로의 Edge Filter / dx -> x방향 미분 차수 / dy -> x방향 미분 차수
    imageDeltaY = cv2.Sobel(image, -1, 0, 1, ksize=3) # y 방향으로의 Edge Filter / dx -> x방향 미분 차수 / dy -> x방향 미분 차수
    return imageDeltaX + imageDeltaY


def imageEdgeScharr(image) :
    imageDeltaX = cv2.Scharr(image, -1, 1, 0) # x 방향으로의 Edge Filter / dx -> x방향 미분 차수 / dy -> x방향 미분 차수 / ksize -> x
    imageDeltaY = cv2.Scharr(image, -1, 0, 1) # y 방향으로의 Edge Filter / dx -> x방향 미분 차수 / dy -> x방향 미분 차수 / ksize -> x
    return imageDeltaX + imageDeltaY


def imageEdgeLaplacianCV(image) :
    return cv2.Laplacian(image, -1) # ksize 지정 가능


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
        
        # Convert into Grayscale Image
        image = convertColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Prewitt Filter
        imagePrewitt = imageEdgePrewitt(image)
        imageShow("Image Prewitt Filter", imagePrewitt)

        # Apply Sobel Filter
        imageSobel = imageEdgeSobel(image)
        imageShow("Image Sobel Filter", imageSobel)

        # Apply Scharr Filter 
        imageScharr = imageEdgeScharr(image)
        imageShow("Image Scharr Filter", imageScharr)

        # Apply Laplacian Filter
        imageLaplacianCV = imageEdgeLaplacianCV(image)
        imageShow("Image Laplacian", imageLaplacianCV)

        # Apply Laplacian Filter (Center Value = -4)
        laplacianMinus4Kernel = np.array([[0, 1, 0],
                                          [1, -4, 1],
                                          [0, 1, 0]], np.float32)

        imageLaplacianFilterMinus4 = imageFiltering(image, laplacianMinus4Kernel)
        imageShow("Image LaplacianFilter (CV=-4)", imageLaplacianFilterMinus4)

        # Apply Laplacian Filter (Center Value = 4)
        laplacianPlus4Kernel = np.array([[0, -1, 0],
                                         [-1, 4, -1],
                                         [0, -1, 0]], np.float32)

        imageLaplacianFilterPlus4 = imageFiltering(image, laplacianPlus4Kernel)
        imageShow("Image LaplacianFilter (CV=4)", imageLaplacianFilterPlus4)

        # Apply Laplacian Filter (Center Value = -8)
        laplacianMinus8Kernel = np.array([[1, 1, 1],
                                          [1, -8, 1],
                                          [1, 1, 1]], np.float32)

        imageLaplacianFilterMinus8 = imageFiltering(image, laplacianMinus8Kernel)
        imageShow("Image LaplacianFilter (CV=-8)", imageLaplacianFilterMinus8)

        # Apply Laplacian Filter (Center Value = 8)
        laplacianPlus8Kernel = np.array([[-1, -1, -1],
                                         [-1, 8, -1],
                                         [-1, -1, -1]], np.float32)

        imageLaplacianFilterPlus8 = imageFiltering(image, laplacianPlus8Kernel)
        imageShow("Image LaplacianFilter (CV=8)", imageLaplacianFilterPlus8)
        
        
        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)