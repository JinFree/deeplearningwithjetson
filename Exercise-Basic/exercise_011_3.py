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
            frameGray = convertColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Auto-Canny Edge Filter
            frameEdge = autoCanny(frameGray, sigma=0.33)
            
            # Set Points
            point1 = (width*0.45, height*0.65)
            point2 = (width*0.55, height*0.65)
            point3 = (width, height*1.0)
            point4 = (0, height*1.0)
            
            # Get ROI Area
            roiCorners = np.array([[point1, point2, point3, point4]], dtype=np.int32)
            frameROI = polyROI(frameEdge, roiCorners)
            
            # Get Lines
            lines = houghLinesP(frameROI, 1, np.pi/180, 40)
            
            # Draw Lane
            frameLane = lineFitting(frame, lines, (0, 0, 255), 5, 30*np.pi/180)

            # Show Processed Frame
            cv2.imshow("Output", frameLane)
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