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
git clone git@github.com:git-emmanuel/13_cuda_matrix.git .

```
or if you are not logged in github:
```bash
git clone https://github.com/git-emmanuel/13_cuda_matrix.git 
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
python -c "..."
```




*This project is licensed under the terms of the MIT license.*

