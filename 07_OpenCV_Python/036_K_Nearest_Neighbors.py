import argparse

import random

import numpy as np
import cv2
import matplotlib.pyplot as plt # pip install matplotlib


#########################################################################################


def main(opt) :
    # Seed 고정
    random.seed(opt.seed), np.random.seed(opt.seed)
    
    # 0 ~ 100 구간 내의 임의의 숫자 2개 샘플링 진행 (총 25x2개)
    trainData = np.random.randint(0, 100, (25,2)).astype(np.float32)
    
    # trainDatat[0]: kick / trainData[1]: kiss / kick > kiss일 경우 1 부여, 아닐 경우 0 부여
    responses = (trainData[:,0] > trainData[:,1]).astype(np.float32)
    
    # 0: Action / 1: Romantic
    action, romantic = trainData[responses==0], trainData[responses==1]
    
    # Action은 파랑 삼각형 / Romantic은 빨강색 동그라미로 표시
    plt.scatter(action[:,0], action[:,1], 80, "b", "^", label="Action")
    plt.scatter(romantic[:,0], romantic[:,1], 80, "r", "o", label="Romantic")
    plt.title("Data Distribution")
    plt.show()
    
    # 0 ~ 100 구간 내의 임의의 숫자 1개 샘플링 진행 (총 1x2개) / 초록색 사각형으로 표시
    newcomer = np.random.randint(0, 100, (1,2)).astype(np.float32)
    plt.scatter(newcomer[:,0], newcomer[:,1], 200, "g", "s", label="new")
    plt.scatter(action[:,0], action[:,1], 80, "b", "^", label="Action")
    plt.scatter(romantic[:,0], romantic[:,1], 80, "r", "o", label="Romantic")
    plt.title("Data Distribution with New Data")
    plt.show()

    # K-Nearest 알고리즘 생성 및 훈련
    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    
    # 결과 예측 (Inferecne)
    ret, results, neighbours, dist = knn.findNearest(newcomer, 7) # K=3
    print(f"< Ret:{ret} | Result:{results} | Neighbours:{neighbours} | Dist:{dist} >")
    
    # 새로운 결과에 화살표로 표시
    annoX, annoY = newcomer.ravel()
    label = "Action" if results == 0 else "Romantic" 
    plt.scatter(newcomer[:,0], newcomer[:,1], 200, "g", "s", label="new")
    plt.scatter(action[:,0], action[:,1], 80, "b", "^", label="Action")
    plt.scatter(romantic[:,0], romantic[:,1], 80, "r", "o", label="Romantic")
    plt.annotate(label, xy=(annoX + 1, annoY+1), xytext=(annoX+5, annoY+10), arrowprops={"color":"black"})
    plt.xlabel("kiss"), plt.ylabel("kick")
    plt.legend(loc="best")
    plt.show()
    

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed number")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)