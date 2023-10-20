# max78_jupyter_training
**Train max7800x from Jupyter**  
The [ai8x-training](https://github.com/MaximIntegratedAI/ai8x-training/tree/pytorch-2.0) is not required!  
Make sure to be on linux!  
Only tested with Nvidia GPUs  
Python 3.8  
Pytorch 2.0.1

## What is included
- MNIST classifier for the MAX78
- QAT (Quantization Aware Training)
- Export KAT
- Export QAT model (.pth.tar), compatible with the [synthesis tool](https://github.com/MaximIntegratedAI/ai8x-synthesis/tree/pytorch-2.0)

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

- Everything should be ready to go! Open the notebook: `train_MNIST.ipynb`

## Summary
- This repo is a fork of the ai8x-training and its purpose is to train the MAX78002 from Jupyter notebook. Instead of using MAX78 training tool.
A picture of QA-training from Jupyter notebook with the distiller:
![QAT max78 notebook](/assets/qat_training.png)
