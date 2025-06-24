import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageRead, imageShow, convertColor, splitImage, histogramEqualization, imageThreshold, mergeImage, imageCopy


#########################################################################################.


def main(opt) :
    # Search for All images
    imageNameList = listdir(opt.imagePath)

    # Load Images
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)
        
        # Load Image
        image = imageRead(imagePath)
        newImage = imageCopy(image)
        imageShow("Image", image)
        
        # 입력된 이미지를 HLS Scale로 바꾼 후 S 채널에 대해 히스토그램 균일화를 수행하세요.
        newImage = convertColor(newImage, cv2.COLOR_BGR2HLS)
        H, L, S = splitImage(newImage)
        Seq = histogramEqualization(S)
        
        # 균일화 된 S 채널에 대해 임계 값 200, 최댓값 255로 cv2.THRESH_BINARY를 수행하세요.
        Seq = imageThreshold(Seq, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
        
        # L 채널에 대해 임계 값 110, 최댓값 128로 cv2.THRESH_BINARY를 수행하세요.
        L = imageThreshold(L, thresh=110, maxval=128, type=cv2.THRESH_BINARY)
        
        # 각 채널을 합친 후 BGR Scale로 바꿔서 확인하세요
        newImageHLS = mergeImage(H, Seq, L)
        newImageHLS = convertColor(newImageHLS, flag=cv2.COLOR_HLS2BGR)
        imageShow("Image HLS Processed", newImageHLS)
        
        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)