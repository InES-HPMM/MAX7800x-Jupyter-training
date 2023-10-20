# max78_jupyter_training
**Train max7800x from Jupyter**  
Make sure to be on linux!  
Only tested with Nvidia GPUs  
Python 3.8  
Pytorch 2.0.1

## Setup
- Setup anaconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html  
- Create the environment and activate it:
```
conda create -n max78-training-jupyter python=3.8
conda activate max78-training-jupyter
```

- Clone the repo and checkout the submodules
```
git clone https://github.com/isztldav/max78_jupyter_training.git
git submodule update --init --recursive
```

- Install requirements
```
pip install -r requirements-cu11.txt
```

- Everything should be ready to go! Open the notebook: train_MNIST.ipynb

