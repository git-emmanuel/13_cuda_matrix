# Project cuda_matrix
Python library for matrix operations

# About the repository
Choice of the gitignore file
https://github.com/github/gitignore

Syntax 
https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax


# Installation steps

01 - Cloning the repository (exluding the project folder)
```bash
mkdir cuda_matrix
cd cuda_matrix
git clone git@github.com:git-emmanuel/13_cuda_matrix.git .

```
or if you are not logged in github:
```bash
git clone https://github.com/git-emmanuel/13_cuda_matrix.git cuda_matrix
```

02 - Create a test conda environment and the necessary packages
*Assuming conda or miniconda is installed*
```bash
conda create --name matrix_test_env python=3.12 -y
conda activate matrix_test_env
pip install -r requirements.txt
```


03 - Run a test code in the project

```bash
python matrix_lib.py
```


04 - Debugging installation of cuda

```bash
conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"
conda install -c conda-forge cuda-python
sudo apt-get install nvidia-cuda-toolkit
conda install numpy time numba matplotlib scipy -y
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-B2775641-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo apt-get install -y cuda-drivers
```
Après l'installation complète, redémarrer l'ordinateur


*This project is licensed under the terms of the MIT license.*

