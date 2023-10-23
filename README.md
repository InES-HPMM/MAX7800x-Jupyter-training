# MAX7800x jupyter training
**Train max7800x from Jupyter notebook**  
The [ai8x-training](https://github.com/MaximIntegratedAI/ai8x-training/tree/pytorch-2.0) is not required!  
Make sure to be on linux!  
Only tested with Nvidia GPUs  
Python 3.8  
Pytorch 2.0.1

# About
For the past several months, I've been deep in the trenches with the [ai8x-training](https://github.com/MaximIntegratedAI/ai8x-training/tree/pytorch-2.0) tool, honing my skills in training various Convolutional Neural Network (CNN) architectures specifically tailored for the MAX78000 and MAX78002 devices. Yet, from the outset, this tool often felt more like a roadblock than an enabler.  

Among the myriad challenges I encountered, the inability to make real-time adjustments, fine-tune models while freezing or unfreezing specific layers, and transferring custom weight sets proved to be major pain points. These limitations stifled the efficiency and flexibility I needed.  

I aim to empower you with the knowledge of how to train the MAX78000 and MAX78002 devices right from your Jupyter notebook. With this approach, you'll break free from the shackles of real-time debugging and fine-tuning woes, ensuring seamless interaction with your neural networks. Let's dive into the nitty-gritty of this process and unlock the full potential of these devices.

## Network Training Completed, What's Next?
It's important to note that this tutorial doesn't substitute the conventional method of synthesizing the network. You'll still need the official tool [ai8x-synthesis](https://github.com/MaximIntegratedAI/ai8x-synthesis/tree/pytorch-2.0) for this critical step.  

The key distinction here is that you now have the flexibility to train your network directly from your Jupyter notebook, bypassing the need to rely on the MAX78 training tool. To see a practical example of the entire process, from training to synthesis, check out the **"How to Get Started"** chapter.

## What is included
- MNIST classifier for the MAX78
- QAT (Quantization Aware Training)
- Export KAT (known-answer test)
- Export QAT model (.pth.tar), compatible with the [synthesis tool](https://github.com/MaximIntegratedAI/ai8x-synthesis/tree/pytorch-2.0)
- Directory structure:

```
.
├── README.md
├── assets
├── data            <-- MNIST dataset storage
├── distiller       <-- distiller submodule, required for QAT
├── max78_modules   <-- max78 python modules
│   ├── ai8x.py
│   └── devices.py
├── requirements-cu11.txt   <-- python requirements
└── train_MNIST.ipynb   <-- example Jupyter notebook
```

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

## How to Get Started

Before diving in, make sure you've completed the setup!

### Training the Network
To train the network, follow these steps:
- Make sure to open the `train_MNIST.ipynb` notebook
- Execute the notebook cells in sequential order.
- The notebook will download the MNIST dataset, train the network, and export the KAT and the QAT model.
- The QAT model should have an accuracy of ~99% on the test set

Take a look at the image below to get a visual idea of what to expect during QAT training:
![QAT max78 notebook](/assets/qat_training.png)

Finally, confirm that the following files have been generated in the `max78_jupyter_training` directory:
- `sample_mnist_2828.npy`
- `qat_class_mnist_checkpoint.pth.tar`

### Synthesizing the Network
- TODO

