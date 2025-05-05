import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, drawCircle


#########################################################################################


def imageAffineTransformation(image, srcPoints, dstPoints, size=None, flags=cv2.INTER_LINEAR) :
    if size is None :
        rows, cols = image.shape[:2]
        size = (cols, rows)
    
    M = cv2.getAffineTransform(srcPoints, dstPoints) # src: 3개의 원본 좌표점 (numpy.ndarray -> shape=(3,2) | np.array([[x1,y1], [x2,y2], [x3,y3]], np.float32)) / 
                                                     # dst: 3개의 결과 좌표점 (numpy.ndarray -> shape=(3,2))
    
    return cv2.warpAffine(image, M, dsize=size, flags=flags) # src: 입력 영상 / M: 2x3 어파인 변환 행렬 (실수형) /
                                                             # dsize: 결과 영상 크기. (w, h) 튜플. (0, 0)이면 src와 같은 크기로 설정 / 
                                                             # flags: 보간법 (기본값 -> cv2.INTER_LINEAR)


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

        # Generate 6 Points for Warp Affine Transformation
        srcPoint1 = [195, 304]
        srcPoint2 = [273, 304]
        srcPoint3 = [280, 351]
        dstPoint1 = [404, 194]
        dstPoint2 = [698, 198]
        dstPoint3 = [698, 386]

        # Aggregate Points
        srcPoints = np.float32([srcPoint1, srcPoint2, srcPoint3])
        dstPoints = np.float32([dstPoint1, dstPoint2, dstPoint3])

        # Show Points
        imagePoint = drawCircle(image, tuple(srcPoint1), 10, (255, 0, 0), -1)
        imagePoint = drawCircle(imagePoint, tuple(srcPoint2), 10, (0, 255, 0), -1)
        imagePoint = drawCircle(imagePoint, tuple(srcPoint3), 10, (0, 0, 255), -1)

        # Apply Warp Affine Transformation
        affineResult1 = imageAffineTransformation(imagePoint, srcPoints, dstPoints)
        affineResult2 = imageAffineTransformation(affineResult1, srcPoints, dstPoints, flags=cv2.WARP_INVERSE_MAP)
        affineResult3 = imageAffineTransformation(affineResult1, dstPoints, srcPoints)

        # Show Results
        imageShow("Image Point", imagePoint)
        imageShow("Affine Result 1", affineResult1)
        imageShow("Affine Result 2", affineResult2)
        imageShow("Affine Result 3", affineResult3)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)