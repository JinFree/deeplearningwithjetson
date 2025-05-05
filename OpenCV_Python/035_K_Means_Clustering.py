import argparse

import random

import numpy as np
import cv2
import matplotlib.pyplot as plt # pip install matplotlib


#########################################################################################


def main(opt) :
    # Seed 고정
    random.seed(opt.seed), np.random.seed(opt.seed)
    
    # 0 ~ 150 구간 내의 임의의 숫자 2개 샘플링 진행 (총 25개)
    sampleA = np.random.randint(0, 150, (25, 2))
    
    # 128 ~ 255 구간 내의 임의의 숫자 2개 샘플링 진행 (총 25개)
    sampleB = np.random.randint(128, 255, (25, 2))
    
    # Sample A와 Sample B를 병합
    data = np.vstack((sampleA, sampleB)).astype(np.float32)
    
    # 데이터 분포 시각화
    plt.scatter(data[:,0], data[:,1], s=100, c="k", marker=".")
    plt.title("Data Distribution")
    plt.show()

    # 반복 중지 요건
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # 평균 클러스터링 적용
    # label -> 결과레이블 / center -> 묶음의중앙점 / 2 -> 묶음개수 / 10 -> 실행횟수
    ret, label, center = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Label에 따라 결과 분류
    red = data[label.ravel()==0]
    blue = data[label.ravel()==1]

    # 결과 출력
    plt.scatter(red[:,0], red[:,1], c="r", marker=".", label="Group A")
    plt.scatter(blue[:,0], blue[:,1], c="b", marker=".", label="Group A")
    
    # 각 그룹의 중앙점 출력
    plt.scatter(center[0,0], center[0,1], s=100, c="r", marker="x", label="Group A Center")
    plt.scatter(center[1,0], center[1,1], s=100, c="b", marker="x", label="Group B Center")
    plt.legend(loc="best")
    plt.title("K-Means Clustering Result")
    plt.show()
    

#########################################################################################


if __name__ == "__main__" :
    # Add Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed number")
    opt = parser.parse_args()
    
    # Excute Main Function
    main(opt)