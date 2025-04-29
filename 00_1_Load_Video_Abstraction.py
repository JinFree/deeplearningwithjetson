import argparse

from os import listdir, makedirs
from os.path import join

from utils import *


#########################################################################################


def main(opt) :
    # Search for All videos
    videoNameList = listdir(opt.videoPath)
    
    # Create Directory for Saving Copied Video
    outputPath = join(opt.savePath, "Saved-Output", "Copied-Video")
    makedirs(outputPath, exist_ok=True)
    
    # Load Videos
    for videoName in videoNameList :
        # Get Full Video Path
        videoPath = join(opt.videoPath, videoName)
        
        # Function-Calling
        processingSingleVideo(videoPath, join(outputPath, videoName))


#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoPath", type=str, required=True, help="path of videos")
    parser.add_argument("--savePath", type=str, required=True, help="path for saving videos")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)