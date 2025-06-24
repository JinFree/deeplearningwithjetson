import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, imageThreshold


#########################################################################################


def imageDilation(image, kernel, iterations) :
    return cv2.dilate(image, kernel=kernel, iterations=iterations) # src: 입력 영상, 바이너리 / kernel: 구조화 요소 커널 / iterations(optional): 팽창 연산 적용 반복 횟수


def imageErosion(image, kernel, iterations) :
    return cv2.erode(image, kernel=kernel, iterations=iterations) # src: 입력 영상, 바이너리 / kernel: 구조화 요소 커널 / iterations(optional): 침식 연산 적용 반복 횟수


def imageMorphologicalGradient(image, iterations=1) :
    kernel = np.ones((3, 3), np.uint8) # 구조화 요소 커널 생성
    dilation = imageDilation(image, kernel, iterations) # 팽창 연산
    erosion = imageErosion(image, kernel, iterations) # 침식 연산
    return dilation-erosion # 결과 값 연산


def imageOpening(image, iterations=1) :
    kernel = np.ones((3, 3), np.uint8) # 구조화 요소 커널 생성
    erosion = imageErosion(image, kernel, iterations) # 침식 연산 
    return imageDilation(erosion, kernel, iterations) # 침식 연산 후 팽창 연산 적용


def imageClosing(image, iterations=1) :
    kernel = np.ones((3, 3), np.uint8) # 구조화 요소 커널 생성
    dilation = imageDilation(image, kernel, iterations) # 팽창 연산
    return imageErosion(dilation, kernel, iterations) # 팽창 연산 후 침식 연산 적용


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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binarize Image
        imageBin = imageThreshold(image, 128, 255, cv2.THRESH_BINARY)
        
        # Generate Kernel for Morphological Operation
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], np.uint8) 
        
        # Apply Morphological Operation
        imageDil = imageDilation(imageBin, kernel, 1)
        imageEro = imageErosion(imageBin, kernel, 1)
        imageOpen = imageOpening(imageBin, 1)
        imageClos = imageClosing(imageBin, 1)
        imageGrad = imageMorphologicalGradient(imageBin)

        # Show Results
        imageShow("Image Threshold", imageBin)
        imageShow("Image Dilation", imageDil)
        imageShow("Image Erosion", imageEro)
        imageShow("Image Opening", imageOpen)
        imageShow("Image Closing", imageClos)
        imageShow("Image Gradient", imageGrad)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)