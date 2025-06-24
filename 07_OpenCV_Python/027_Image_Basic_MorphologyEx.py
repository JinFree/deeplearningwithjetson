import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageRead, imageShow, imageThreshold


#########################################################################################


def imageMorphologyKernel(flag=cv2.MORPH_RECT, size=5) :
    return cv2.getStructuringElement(flag, (size, size)) # shape: 구조화 요소 커널 모양 (cv2.MORPH_RECT: 사각형 | cv2.MORPH_EPLIPSE: 타원형 | cv2.MORPH_CROSS: 십자형) / ksize: 커널 크기


def imageMorphologyEx(image, op, kernel, iterations=1) :
    return cv2.morphologyEx(image, op=op, kernel=kernel, iterations=iterations) # src: 입력 영상 / op: 모폴로지 연산 종류 (cv2.MORPH_OPEN: 열림 연산 | cv2.MORPH_CLOSE: 닫힘 연산 | cv2.MORPH_GRADIENT: 그레디언트 연산 | cv2.MORPH_TOPHAT: 탑햇 연산 | cv2.MORPH_BLACKHAT: 블랙햇 연산) /
                                                                                # kernel: 구조화 요소 커널 / iteration(optional): 연산 반복 횟수


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

        # Binarize image
        imageBin = imageThreshold(image, 128, 255, cv2.THRESH_BINARY)
        
        # Get Kernels for Morphological Operation
        kernelRect = imageMorphologyKernel(cv2.MORPH_RECT, 5)
        kernelEllipse = imageMorphologyKernel(cv2.MORPH_ELLIPSE, 5)
        kernelCross = imageMorphologyKernel(cv2.MORPH_CROSS, 5)

        # Apply Opening Morphological Operation
        imageOpenRect = imageMorphologyEx(imageBin, cv2.MORPH_OPEN, kernelRect) # 열림 연산
        imageOpenEllipse = imageMorphologyEx(imageBin, cv2.MORPH_OPEN, kernelEllipse) # 열림 연산
        imageOpenCross = imageMorphologyEx(imageBin, cv2.MORPH_OPEN, kernelCross) # 열림 연산

        # Show Results
        imageShow("Image Threshold", imageBin)
        imageShow("MORPH_RECT Kernel", kernelRect)
        imageShow("Image Opening (Rect)", imageOpenRect)
        imageShow("MORPH_ELLIPSE Kernel", kernelEllipse)
        imageShow("Image Opening (Ellipse)", imageOpenEllipse)
        imageShow("MORPH_CROSS Kernel", kernelCross)
        imageShow("Image Opening (Cross)", imageOpenCross)

        cv2.destroyAllWindows()

        # Apply Closing Morphological Operation
        imageCloseRect = imageMorphologyEx(imageBin, cv2.MORPH_CLOSE , kernelRect) # 닫힘 연산
        imageCloseEllipse = imageMorphologyEx(imageBin, cv2.MORPH_CLOSE, kernelEllipse) # 닫힘 연산 
        imageCloseCross = imageMorphologyEx(imageBin, cv2.MORPH_CLOSE, kernelCross) # 닫힘 연산

        # Show Results
        imageShow("Image Threshold", imageBin)
        imageShow("MORPH_RECT Kernel", kernelRect)
        imageShow("Image Closing (Rect)", imageCloseRect)
        imageShow("MORPH_ELLIPSE Kernel", kernelEllipse)
        imageShow("image Closing (Ellipse)", imageCloseEllipse)
        imageShow("MORPH_CROSS Kernel", kernelCross)
        imageShow("image Closing (Cross)", imageCloseCross)

        cv2.destroyAllWindows()
        
        # Apply Gradient Morphological Operation
        imageGradientRect = imageMorphologyEx(imageBin, cv2.MORPH_GRADIENT , kernelRect) # 그레디언트 연산
        imageGradientEllipse = imageMorphologyEx(imageBin, cv2.MORPH_GRADIENT , kernelEllipse) # 그레디언트 연산
        imageGradientCross = imageMorphologyEx(imageBin, cv2.MORPH_GRADIENT , kernelCross) # 그레디언트 연산

        imageShow("Image Threshold", imageBin)
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
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)