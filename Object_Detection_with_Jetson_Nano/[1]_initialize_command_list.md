- 도커 권한을 수정합니다.
```
sudo usermod -aG docker $USER
```

- 불필요한 패키지를 삭제합니다.
```
sudo apt remove --purge libreoffice*
sudo apt clean
sudo apt autoremove
```

- 몇가지 패키지를 설치합니다.
```
sudo apt update
sudo apt install -y apt-utils nvidia-jetpack wget curl git vim tmux python3-pip
sudo -H python3 -m pip install -U jetson-stats
```

- CUDA 경로를 ~/.bashrc에 등록합니다.
```
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc
echo "export PATH=\$PATH:/usr/local/cuda/bin" >> ~/.bashrc
echo "set -g mouse on" >> ~/.tmux.conf
```

- pycuda를 설치합니다.
```
source ~/.bashrc
python3 -m pip install pycuda
```

- 작업 경로를 생성하고 데이터를 다운로드 받습니다.
```
cd ~
mkdir workspace
cd workspace
git clone https://github.com/JinFree/deeplearningwithjetson
wget https://github.com/JinFree/OpenCV_in_Ubuntu/raw/master/Data/Lane_Detection_Videos/challenge.mp4
mkdir images
cd images
wget https://ultralytics.com/images/zidane.jpg
wget https://ultralytics.com/images/bus.jpg
```

- Jetson nano의 swap 메모리 공간을 추가합니다.
```
sudo fallocate -l 4G /swapfile
ls -lh /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo cp /etc/fstab /etc/fstab.bak
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

- Jetson에서 다양한 Docker images를 쉽게 사용하기 위한 jetson-containers를 설치합니다.
```
cd ~
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```

- /media/nvidia/L4T-README/README-vnc.txt 의 설명에 따라 vnc 환경을 준비합니다.
```
sudo apt update
sudo apt install vino
mkdir -p ~/.config/autostart
cp /usr/share/applications/vino-server.desktop ~/.config/autostart
gsettings set org.gnome.Vino prompt-enabled false
gsettings set org.gnome.Vino require-encryption false
gsettings set org.gnome.Vino authentication-methods "['vnc']"
gsettings set org.gnome.Vino vnc-password $(echo -n 'nvidia'|base64)
```

- 재부팅합니다.