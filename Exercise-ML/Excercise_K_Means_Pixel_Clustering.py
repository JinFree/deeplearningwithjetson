import argparse

from os import listdir
from os.path import join

import random

import numpy as np
import cv2

from utils import imageRead, imageShow


#########################################################################################


def main(opt) :
    # Seed 고정
    random.seed(opt.seed), np.random.seed(opt.seed)
    
    # Search for All images
    imageNameList = listdir(opt.imagePath)

    # Load Images
    for imageName in imageNameList :
        # Get Full Image Path
        imagePath = join(opt.imagePath, imageName)
        
        # Load Image
        image = imageRead(imagePath)
    
        # Flatten Image Spatial-wise
        # 코드를 입력해주세요.
        
        # 반복 중지 요건
        # 코드를 입력해주세요.

        # 평균 클러스터링 적용
        # 코드를 입력해주세요.

        # Get Centers of Different Clusters
        centers = centers.astype(np.uint8)
        
        # Reshape Image
        dst = centers[bestLabels].reshape(image.shape)
        
        # Show Results
        imageShow("Image Original", image)
        imageShow("K-Means Clustered Image", dst)
        
        cv2.destroyAllWindows()
    

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed number")
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)