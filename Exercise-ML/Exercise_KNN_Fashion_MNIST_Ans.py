import argparse

from os import listdir
from os.path import join

import random

import numpy as np
import cv2

from tqdm import tqdm

from utils import imageShow, imageResize


#########################################################################################


def loadTrainData(imagePath, labelPath) :
    with open(imagePath, "rb") as imageData :
        images = np.frombuffer(imageData.read(), dtype=np.uint8, offset=16)
        print("Image Loaded!")
    with open(labelPath, "rb") as labelData :
        labels = np.frombuffer(labelData.read(), dtype=np.uint8, offset=8)
        print("Label Loaded!")
    return images.reshape(-1, 784), labels # images.reshape(-1, 784) -> Image Flattening Process


#########################################################################################

def main(opt) :
    # Seed 고정
    random.seed(opt.seed), np.random.seed(opt.seed)
    
    # Load Dataset
    trainX, trainY = loadTrainData(join(opt.dataPath, "train-images-idx3-ubyte"), join(opt.dataPath, "train-labels-idx1-ubyte"))
    testX, testY = loadTrainData(join(opt.dataPath, "t10k-images-idx3-ubyte"), join(opt.dataPath, "t10k-labels-idx1-ubyte"))
    
    # Create Dictionary for Adding label
    labelDict = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}
    
    # Train KNN Model
    knn = cv2.ml.KNearest_create()
    retval = knn.train(trainX.astype(np.float32), cv2.ml.ROW_SAMPLE, trainY.astype(np.int32))
    
    # Inference Process
    retval, results, neighborResponses, dist = knn.findNearest(testX.astype(np.float32), k=7)
    
    # Compute Accuracy
    accuracy = 0
    with tqdm(total=len(testY)) as pBar :
        for i in range(len(testY)) :
            if results[i][0].astype(np.uint8) == testY[i] :
                accuracy += 1
            pBar.update()
    accuracy = accuracy*100/len(testY)
    print(f"Accuracy : {accuracy}%")
    
    # Visualize Result
    count = 0
    for idx, result in enumerate(results):
        print("Index : {}".format(idx))
        print("예측값 : {}".format(labelDict[int(result)]))
        print("실제값 : {}".format(labelDict[testY[idx]]))
        imageShow("Images", imageResize(testX[idx].reshape(28, 28, 1), (256, 256), interpolation=cv2.INTER_CUBIC))
        count += 1
        cv2.destroyAllWindows()
        
        if count >= 20 :
            break

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed number")
    parser.add_argument("--dataPath", type=str, required=True, help="path of images")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)