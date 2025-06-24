import argparse

from os import listdir
from os.path import join

import cv2

from utils import *


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
        rows, cols = image.shape[:2]
        size = (cols, rows)
        imageShow("Image", image)
        
        # No-Change Matrix
        M = np.array([[1,0,0], [0,1,0]], dtype=np.float32)
        newImage = cv2.warpAffine(image, M, size, flags=cv2.INTER_LINEAR)
        imageShow("No Change", newImage)
        
        # Translation Matrix
        delta = 50
        M = np.array([[1,0,delta],[0,1,delta]], dtype=np.float32)
        newImage = cv2.warpAffine(image, M, size, flags=cv2.INTER_LINEAR)
        imageShow("Translation", newImage)
        
        # Scaling Matix
        scale = 1.5
        M = np.array([[scale,0,0], [0,scale,0]], dtype=np.float32)
        newImage = cv2.warpAffine(image, M, size, flags=cv2.INTER_LINEAR)
        imageShow("Scaling", newImage)
        
        # Rotation Matrix
        angle = np.pi/4
        M = np.array([[np.cos(angle),-np.sin(angle),0], [np.sin(angle),np.cos(angle),0]], dtype=np.float32)
        newImage = cv2.warpAffine(image, M, size, flags=cv2.INTER_LINEAR)
        imageShow("Rotation", newImage)
        
        cv2.destroyAllWindows()
        

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)