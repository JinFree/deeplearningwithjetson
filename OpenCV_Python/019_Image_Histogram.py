import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, convertColor


#########################################################################################


def computeHist(image, mask=None) :
    bins = np.arange(256).reshape(256,1) # Sample의 구간을 설정
    
    if len(image.shape) == 2 :
        h = np.zeros((300, 256, 1))
        histItem = cv2.calcHist([image], [0], None, [256], [0,255]) # cv2.calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
        cv2.normalize(histItem, histItem, 0, 255, cv2.NORM_MINMAX) # 정규화를 적용하여, 이미지 처리를 진행
        hist = np.int32(np.around(histItem))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, 255)
        
    elif len(image.shape) == 3 :
        h = np.zeros((300, 256, 3))
        color = [(255,0,0), (0,255,0), (0,0,255)] 
        for ch, col in enumerate(color) :
            histItem = cv2.calcHist([image], [ch], None, [256], [0,255]) # cv2.calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
            cv2.normalize(histItem, histItem, 0, 255, cv2.NORM_MINMAX) # 정규화를 적용하여, 이미지 처리를 진행
            hist = np.int32(np.around(histItem)) 
            pts = np.column_stack((bins, hist)) 
            cv2.polylines(h, [pts], False, col)
    
    return np.flipud(h)


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

        # Plot Histogram of RGB Image
        imageHist = computeHist(image)
        imageShow("Image Hist", imageHist)

        # Plot Histogram of Grayscale Image
        imageGray = convertColor(image, cv2.COLOR_BGR2GRAY)
        imageShow("Image Gray", imageGray)
        imageGrayHist = computeHist(imageGray)
        imageShow("Image GrayHist", imageGrayHist)

        # Plot Histogram of HSV Image
        imageHSV = convertColor(image, cv2.COLOR_BGR2HSV)
        imageShow("Image HSV", imageHSV)
        imageHSVHist = computeHist(imageHSV)
        imageShow("Image HSVHist", imageHSVHist)

        # Plot Histogram of HLS Image
        imageHLS = convertColor(image, cv2.COLOR_BGR2HLS)
        imageShow("Image HLS", imageHLS)
        imageHLSHist = computeHist(imageHLS)
        imageShow("Image HLSHist", imageHLSHist)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)