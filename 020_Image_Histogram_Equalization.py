import argparse

from os import listdir
from os.path import join

import cv2

from utils import splitImage, mergeImage, imageRead, imageShow, convertColor


#########################################################################################


def histogramEqualization(image) :
    if len(image.shape) == 2 :
        return cv2.equalizeHist(image) # 채널에 대한 Equalization 진행
    else :
        ch1, ch2, ch3 = splitImage(image) # 채널 분리
        ch1Eq = cv2.equalizeHist(ch1) # 각 채널에 대한 Equalization 진행
        ch2Eq = cv2.equalizeHist(ch2) # 각 채널에 대한 Equalization 진행
        ch3Eq = cv2.equalizeHist(ch3) # 각 채널에 대한 Equalization 진행
        return mergeImage(ch1Eq, ch2Eq, ch3Eq)


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
        
        # Apply Histogram Equalization
        imageEq = histogramEqualization(image)
        imageShow("Image Equalized (BGR)", imageEq)

        # Apply Histogram Equalization to Grayscale Image
        imageGray = convertColor(image, cv2.COLOR_BGR2GRAY)
        imageShow("imageGray", imageGray)
        image_GRAYEqualized = histogramEqualization(imageGray)
        imageShow("Image Equalized (Grayscale)", image_GRAYEqualized)

        # Apply Histogram Equalization to HSV Image
        imageHSV = convertColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = splitImage(imageHSV)
        sEq = histogramEqualization(s)
        vEq = histogramEqualization(v)
        imageHSeqVeq = mergeImage(h, sEq, vEq)
        imageHSVeq = mergeImage(h, s, vEq)
        imageHSeqV = mergeImage(h, sEq, v)
        imageHSeqVeq2BGR = convertColor(imageHSeqVeq, cv2.COLOR_HSV2BGR)
        imageHSVeq2BGR = convertColor(imageHSVeq, cv2.COLOR_HSV2BGR)
        imageHSeqV2BGR = convertColor(imageHSeqV, cv2.COLOR_HSV2BGR)
        imageShow("Image", image)
        imageShow("Image Equalized (S-Eq + V-Eq)", imageHSeqVeq2BGR)
        imageShow("Image Equalized (V-Eq)", imageHSVeq2BGR)
        imageShow("Image Equalized (S-Eq)", imageHSeqV2BGR)

        # Apply Histogram Equalization to HLS Image
        imageHLS = convertColor(image, cv2.COLOR_BGR2HLS)
        h, l, s = splitImage(imageHLS)
        lEq = histogramEqualization(l)
        sEq = histogramEqualization(s)
        imageHLeqSeq = mergeImage(h, lEq, sEq)
        imageHLSeq = mergeImage(h, l, sEq)
        imageHLeqS = mergeImage(h, lEq, s)
        imageHLeqSeq2BGR = convertColor(imageHLeqSeq, cv2.COLOR_HLS2BGR)
        imageHLSeq2BGR = convertColor(imageHLSeq, cv2.COLOR_HLS2BGR)
        imageHLeqS2BGR = convertColor(imageHLeqS, cv2.COLOR_HLS2BGR)
        imageShow("Image", image)
        imageShow("Image Equalized (L-Eq + S-Eq)", imageHLeqSeq2BGR)
        imageShow("Image Equalized (S-Eq)", imageHLSeq2BGR)
        imageShow("Image Equalized (L-Eq)", imageHLeqS2BGR)

        cv2.destroyAllWindows()
        

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)
