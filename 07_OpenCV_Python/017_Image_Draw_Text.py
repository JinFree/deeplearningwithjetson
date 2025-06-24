import argparse

from os import listdir
from os.path import join

import cv2

from utils import imageCopy, imageRead, imageShow, drawCircle


#########################################################################################


def drawText(image, text, point=(10, 10), font=cv2.FONT_HERSHEY_PLAIN, fontScale=2.0, color=(255,255,255), thickness=3, lineType=cv2.LINE_AA) : # Line 및 Font Type -> https://076923.github.io/posts/Python-opencv-18/
    result = imageCopy(image)
    return cv2.putText(result, text, point, font, fontScale, color, thickness, lineType)


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

        # Draw Text on Image
        imageText1 = drawText(image, "Cute Doll", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 255, 255), 3)
        imageText1 = drawCircle(imageText1, (10, 50), 5, (0, 0, 255), -1)
        imageShow("imageText1", imageText1)

        # Change Fonts
        imageText2 = drawText(image, "cv2.FONT_HERSHEY_SIMPLEX", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
        imageText2 = drawText(imageText2, "cv2.FONT_HERSHEY_PLAIN", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        imageText2 = drawText(imageText2, "cv2.FONT_HERSHEY_DUPLEX", (10, 150), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        imageText2 = drawText(imageText2, "cv2.FONT_HERSHEY_COMPLEX", (10, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 1)
        imageText2 = drawText(imageText2, "cv2.FONT_HERSHEY_TRIPLEX", (10, 250), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
        imageText2 = drawText(imageText2, "cv2.FONT_HERSHEY_COMPLEX_SMALL", (10, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 1)
        imageText2 = drawText(imageText2, "cv2.FONT_HERSHEY_SCRIPT_SIMPLEX", (10, 350), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (255, 255, 255), 1)
        imageText2 = drawText(imageText2, "cv2.FONT_HERSHEY_SCRIPT_COMPLEX", (10, 400), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, (255, 255, 255), 1)
        imageShow("imageText2", imageText2)

        cv2.destroyAllWindows()


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagePath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)