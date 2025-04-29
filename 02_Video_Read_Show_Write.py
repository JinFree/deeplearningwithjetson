import argparse

from os import listdir, makedirs
from os.path import join

import cv2

from tqdm import tqdm

from utils import imageProcessing


#########################################################################################


def Video(videoPath, savePath=None) :
    # Load Video
    cap = cv2.VideoCapture(videoPath)
    
    if cap.isOpened() :
        print("Video Opened")
    else :
        print("Video Not Opened")
        print("Program Abort")
        exit()
    
    # Get Video Parameters
    fps = cap.get(cv2.CAP_PROP_FPS) # Frame Per Second
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 전체 프레임 수
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # 영상의 Width 크기 (가로)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 영상의 Height 크기 (세로)
    out = None
    
    # Show Features of the Input Video
    print(f"< FPS : {fps} || Video Size : {width} x {height} >")
    
    if savePath is not None :
        fourcc = cv2.VideoWriter_fourcc(*"MPEG") # 4-문자 코드, four character code 지정 [*'PIM1' / *'MJPG' / *'DIVX' / *'XVID' / *'MPEG' / *'X264' 코덱 지원]
        out = cv2.VideoWriter(savePath, fourcc, fps, (width, height), True)
    
    # Play Video
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    
    # TQDM Bar
    with tqdm(total=frameCount) as pBar :
        while cap.isOpened() :
            # Capture frame-by-frame
            ret, frame = cap.read() # -> ret : retrieval (영상을 획득하면 True)
            
            if ret :
                # Our operations on the frame come here
                output = imageProcessing(frame)
                
                if out is not None:
                    # Write frame-by-frame
                    out.write(output)
                
                # Display the resulting frame
                cv2.imshow("Input", frame)
                cv2.imshow("Output", output)
            else :
                break
            
            # waitKey(int(1000.0/fps)) for matching fps of video
            if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q') :
                break
            
            # Update TQDM Bar
            pBar.set_description(f"< Video Size : {frame.shape} >")
            pBar.update()
            
    # When Everything Is Done, Release the Capture
    cap.release()
    if out is not None:
        out.release()

    cv2.destroyAllWindows()


#########################################################################################


def main(opt) :
    # Search for All Videos
    videoNameList = listdir(opt.videoPath)
    
    # Create Directory for Saving Copied Video
    outputPath = join(opt.savePath, "Saved-Output", "Copied-Video")
    makedirs(outputPath, exist_ok=True)

    # Load Images
    for videoName in videoNameList :
        # Get Full Image Path
        videoPath = join(opt.videoPath, videoName)
        
        # Fuction-Calling
        Video(videoPath, join(outputPath, videoName))


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoPath", type=str, required=True, help="path of videos")
    parser.add_argument("--savePath", type=str, required=True, help="path for saving videos")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)