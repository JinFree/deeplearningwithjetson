import argparse

from os import listdir
from os.path import join

import cv2

from utils import *


#########################################################################################


def Video(videoPath) :
    # Load Video
    cap = cv2.VideoCapture(videoPath)
    
    # Error-Handling
    if cap.isOpened() :
        print("Video Opened")
    else :
        print("Video Not Opened")
        print("Program Abort")
        exit()
    
    # Get Video Parameters
    fps = cap.get(cv2.CAP_PROP_FPS) # Frame Per Second
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 영상의 Width 크기 (가로)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 영상의 Height 크기 (세로)
    
    # Play Video
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    
    currentFrame = 0
    
    while cap.isOpened() :
        # Capture Frame-by-Frame
        ret, frame = cap.read() # -> ret : retrieval (영상을 획득하면 True)
        
        if ret :
            # Convert BGR to Grayscale
            frame = convertColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Auto-Canny Edge Filter
            frame = autoCanny(frame, sigma=0.33)

            # Show Processed Frame
            cv2.imshow("Output", frame)
        else :
            break
        
        if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q') :
            break
        
        currentFrame += 1

    # When Everything Is Done, Release the Capture
    cap.release()

    cv2.destroyAllWindows()


#########################################################################################


def main(opt) :
    # Search for All Videos
    videoNameList = listdir(opt.videoPath)

    # Load Images
    for videoName in videoNameList :
        # Get Full Image Path
        videoPath = join(opt.videoPath, videoName)
        
        # Fuction-Calling
        Video(videoPath)


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoPath", type=str, required=True, help="path of videos")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)