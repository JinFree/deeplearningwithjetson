Tensorrtx를 준비합니다.
```
cd ~/workspace
git clone https://github.com/wang-xinyu/tensorrtx
```

Jetson-containers의 ML 이미지를 실행합니다.
```
docker run -it --privileged --rm --ipc=host --network=host --runtime=nvidia -v /home/nvidia/workspace:/workspace dustynv/l4t-ml:r32.7.1
```
- 이후의 실습에서 파일이 새로 써지지 않는 권한 문제가 발생하는 경우 새 터미널에서 아래 명령어를 입력합니다.
  ```
  sudo chown -R nvidia ~/workspace
  ```
- 아래와 같은 로그를 확인하면 JupyterLab에 접근할 수 있게 됩니다.
  ```
  allow 10 sec for JupyterLab to start @ http://IP:8888 (password nvidia)
  JupterLab logging location:  /var/log/jupyter.log  (inside the container)
  ```

아래 명령어는 l4t-ml 도커 내부에서 실행합니다.
- 경로를 이동한 후 환경을 준비합니다.
  ```
  cd /workspace/tensorrtx/yolov7
  python3 -m pip install tqdm matplotlib seaborn
  git clone -b v0.1 https://github.com/WongKinYiu/yolov7
  wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
  cd yolov7
  cp ../gen_wts.py .
  ```

- yolov7-tiny.pt를 engine으로 빌드하기 위한 wts 파일을 생성합니다.
  ```
  python3 gen_wts.py -w ../yolov7-tiny.pt -o ../yolov7-tiny.wts
  ```

아래 명령어는 도커 바깥, 호스트에서 실행합니다.
- 경로를 이동한 후 환경을 준비합니다.
  ```
  cd /home/nvidia/workspace/tensorrtx/yolov7
  sudo chown -R nvidia /home/nvidia/workspace
  mkdir build
  cd build/
  ```

- trt engine을 빌드하고 추론하기 위한 tensorrtx를 빌드합니다.
  ```
  cmake ..
  make -j 2
  ```

- engine을 빌드합니다.
  ```
  ./yolov7 -s ../yolov7-tiny.wts yolov7-tiny.engine t
  ```

- 빌드한 엔진으로 이미지를 추론합니다.
  ```
  ./yolov7 -d yolov7-tiny.engine /home/nvidia/workspace/images
  ```

- 파이썬 스크립트로 이미지를 추론합니다.
  ```
  cd /home/nvidia/workspace/tensorrtx/yolov7
  python3 yolov7_trt.py build/yolov7-tiny.engine build/libmyplugins.so
  ```