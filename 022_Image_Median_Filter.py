import argparse

from os import listdir
from os.path import join

import cv2
import numpy as np

from utils import imageRead, imageCopy


#########################################################################################


def imageMedianBlur(image, size) :
    ksize = (size+1)*2 - 1
    return cv2.medianBlur(image, ksize)


def nothing(x) :
    pass


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
        cv2.namedWindow("Image", cv2.WINDOW_GUI_EXPANDED)
        
        # Add Noise
        noise = np.random.normal(scale=opt.sigma/255, size=image.shape[:2])
        noisyImage = imageCopy(image)/255
        noisyImage[:,:,0] += noise
        noisyImage[:,:,1] += noise
        noisyImage[:,:,2] += noise
        noisyImage = (np.clip(noisyImage, 0, 1)*255).astype(np.uint8)

        # Set Track Bar
        cv2.createTrackbar("Blur Size", "Image", 0, 10, nothing)

        switch = "0:OFF\n1:On"
        cv2.createTrackbar(switch, "Image", 1, 1, nothing)

        while True :
            # Show Loaded Image
            cv2.imshow("Image", image)

            # ECS Key를 사용하여 프로그램 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            size = cv2.getTrackbarPos("Blur Size", "Image") # Trackbar 명칭 / Trackbar가 등록된 Named Window
            s = cv2.getTrackbarPos(switch, "Image") # Trackbar 명칭 / Trackbar가 등록된 Named Window

            if s == 1 :
                image = imageMedianBlur(noisyImage, size) # Apply Median Filter
            else:
                image = noisyImage
            
        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    parser.add_argument("--sigma", type=int, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)