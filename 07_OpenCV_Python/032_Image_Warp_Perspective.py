import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageShow, drawCircle


#########################################################################################


def imagePerspectiveTransformation(image, srcPoints, dstPoints, size=None, flags=cv2.INTER_LANCZOS4) :
    if size is None :
        rows, cols = image.shape[:2]
        size = (cols, rows)
    
    M = cv2.getPerspectiveTransform(srcPoints, dstPoints) # src: 4개의 원본 좌표점 (numpy.ndarray -> shape=(4,2) | np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], np.float32)) / 
                                                          # dst: 4개의 결과 좌표점 (numpy.ndarray -> shape=(4,2))
    
    return cv2.warpPerspective(image, M, dsize=size, flags=flags) # src: 입력 영상 / M: 3x3 어파인 변환 행렬 (실수형) /
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

        # Get Image Shape
        height, width = image.shape[:2]

        # Generate 8 Points for Warp Affine Transformation
        srcPoint1 = [int(width*0.35), int(height*0.65)]
        srcPoint2 = [int(width*0.65), int(height*0.65)]
        srcPoint3 = [width, height]
        srcPoint4 = [0, height]
        dstPoint1 = [int(width*0.1), 0]
        dstPoint2 = [int(width*0.9), 0]
        dstPoint3 = [int(width*0.9), height]
        dstPoint4 = [int(width*0.1), height]

        # Aggregate Points
        srcPoints = np.float32([srcPoint1, srcPoint2, srcPoint3, srcPoint4])
        dstPoints = np.float32([dstPoint1, dstPoint2, dstPoint3, dstPoint4])

        # Show Points 
        roadPoint = drawCircle(image, tuple(srcPoint1), 10, (255, 0, 0), -1)
        roadPoint = drawCircle(roadPoint, tuple(srcPoint2), 10, (0, 255, 0), -1)
        roadPoint = drawCircle(roadPoint, tuple(srcPoint3), 10, (0, 0, 255), -1)
        roadPoint = drawCircle(roadPoint, tuple(srcPoint4), 10, (255, 255, 0), -1)

        # Apply Warp Affine Transformation
        affineResult1 = imagePerspectiveTransformation(roadPoint, srcPoints, dstPoints)
        affineResult2 = imagePerspectiveTransformation(affineResult1, srcPoints, dstPoints, flags=cv2.WARP_INVERSE_MAP)
        affineResult3 = imagePerspectiveTransformation(affineResult1, dstPoints, srcPoints)

        # Show Results
        imageShow("Image", image)
        imageShow("Road Point", roadPoint)
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