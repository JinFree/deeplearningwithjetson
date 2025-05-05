import argparse

from os import listdir, makedirs
from os.path import join

import cv2
import numpy as np


#########################################################################################


def imageRead(imagePath, flag=cv2.IMREAD_UNCHANGED) :
    # If No Korea in Path
    # image = cv2.imread(imagePath, flag)
    
    # If Korean in Path
    imageArray = np.fromfile(imagePath, np.uint8)
    image = cv2.imdecode(imageArray, flag)
    
    # Deal with Error
    if image is not None :
        print("Image Opened")
        return image
    else :
        print("Image Not Opened")
        print("Program Abort")
        exit()


def imageShow(imageName, image, flag=cv2.WINDOW_GUI_EXPANDED) :
    # Show Image using GUI
    cv2.namedWindow(imageName, flag)
    cv2.imshow(imageName, image)
    cv2.waitKey()


def imageWrite(imageName, image) :
    # Save Image
    return cv2.imwrite(imageName, image)


#########################################################################################


def main(opt) :
    # Search for All Images
    imageNameList = listdir(opt.imagePath)
    
    # Create Directory for Saving Copied Image
    outputPath = join(opt.savePath, "Grayscale-Image")
    makedirs(outputPath, exist_ok=True)
    
    # Load Images
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)

        # cv2.IMREAD_COLOR
        # Blue - Green - Red 순서의 3채널 이미지로 열림
        imageColor = imageRead(imagePath, cv2.IMREAD_COLOR)
        print(type(imageColor))

        # cv2.IMREAD_GRAYSCALE
        # 회색조 1채널로 열림
        imageGray = imageRead(imagePath, cv2.IMREAD_GRAYSCALE)

        # cv2.IMREAD_UNCHANGED
        # 1채널 회색조 이미지는 1채널로 열리고, 3채널 컬러 이미지는 3채널 BGR로 열림
        imageSrc = imageRead(imagePath, cv2.IMREAD_UNCHANGED)

        # cv2.WINDOW_NORMAL: 화면 크기 전환 가능 및 비율 유지
        imageShow("Image-Source (cv2.WINDOW_NORMAL)", imageSrc, cv2.WINDOW_NORMAL)
        
        # cv2.WINDOW_NORMAL: 화면 크기 전환 가능 및 비율 유지
        imageShow("Image-Color (cv2.WINDOW_NORMAL)", imageColor, cv2.WINDOW_NORMAL)

        # cv2.WINDOW_AUTOSIZE : 화면 크기 전환 불가능 및 비율 유지
        imageShow("Image-Color (cv2.WINDOW_AUTOSIZE)", imageColor, cv2.WINDOW_AUTOSIZE)

        # cv2.WINDOW_FREERATIO : 화면 크기 전환 가능 및 비율 유지 안됨
        imageShow("Image-Color (cv2.WINDOW_FREERATIO)", imageColor, cv2.WINDOW_FREERATIO)

        # cv2.WINDOW_GUI_NORMAL : 화면 크기 전환 가능 및 비율 유지
        imageShow("Image-Color (cv2.WINDOW_GUI_NORMAL)", imageColor, cv2.WINDOW_GUI_NORMAL)

        # cv2.WINDOW_GUI_EXPANDED : 화면 크기 전환 가능 및 비율 유지
        imageShow("Image-Color (cv2.WINDOW_GUI_EXPANDED)", imageColor, cv2.WINDOW_GUI_EXPANDED)

        # Save Result
        imageWrite(join(outputPath, imageName), imageGray)


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    parser.add_argument("--savePath", type=str, required=True, help="path for saving images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)