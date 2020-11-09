echo "Installing CUDA 8"
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
rm cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
apt-key add /var/cuda-repo-8-0-local-ga2/7fa2af80.pub
apt-get update
apt-get install cuda-8-0
pip install torch==0.4.0
sudo apt-get update && \
sudo apt-get install build-essential software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y 
sudo apt-get update
sudo apt-get install gcc-5 g++-5 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50 --slave /usr/bin/g++ g++ /usr/bin/g++-5 
sudo apt-get install gcc-5 g++-5 g++-5-multilib gfortran-5
sudo update-alternatives --config gcc

echo "Downloading planercnn"

git clone https://github.com/NVlabs/planercnn
cd planercnn/nms/src/cuda/
echo "Compiling nms"
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ../../
python build.py
cd ..
cd roialign/roi_align/src/cuda/
echo "Compiling roi_align"
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_60
cd ../../
python build.py
cd ../../
# wget https://www.dropbox.com/s/yjcg6s57n581sk0/checkpoint.zip?dl=0
gdown --id 1o2wZG0swF-HImZbQGPC7cHONkCThFVQZ -O checkpoint.zip?dl=0
mkdir checkpoint
mv "checkpoint.zip?dl=0" "planercnn_refine.zip"
mv planercnn_refine.zip checkpoint/
cd checkpoint/
unzip planercnn_refine.zip
rm planercnn_refine.zip
cd ..
pip install -r requirements.txt
echo "Running Tests"
python evaluate.py --methods=f --suffix=warping_refine --dataset=inference --customDataFolder=example_images
echo "Setup Complete"