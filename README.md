# MAX7800x jupyter training
In this guide, we'll walk you through **training MAX7800x directly from a Jupyter notebook**. The best part? You won't need the [ai8x-training](https://github.com/MaximIntegratedAI/ai8x-training/tree/pytorch-2.0) tool for this process.

Please ensure you are using a Linux environment, as this guide is tailored for Linux users. Additionally, this method has been tested exclusively with Nvidia GPUs. Make sure you have the following prerequisites in place:

- Python 3.8
- PyTorch 2.0.1

Let's dive into the world of training MAX7800x, right from the comfort of your Jupyter notebook.

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
- **IMPORTANT**: It's crucial to ensure that you have a functional installation of the [ai8x-synthesis](https://github.com/MaximIntegratedAI/ai8x-synthesis/tree/pytorch-2.0) tool. If you haven't done so yet, please refer to the official GitHub repository and follow the provided instructions for installation.

- **Note**: The following steps are not unique but are standard for synthesizing any trained network using the ai8x-synthesis tool.

#### Prepare the KAT
- Copy the `sample_mnist_2828.npy` file to the `ai8x-synthesis/tests` directory
#### Prepare the QAT model
- Make a subfolder in the `ai8x-synthesis/` directory, for example `ai8x-synthesis/custom-mnist`
- Copy the `qat_class_mnist_checkpoint.pth.tar` file to the `custom-mnist` directory
- Quantize the network: `python quantize.py custom-mnist/qat_class_mnist_checkpoint.pth.tar custom-mnist/qat_class_mnist_checkpoint_q8.pth.tar --device MAX78002 -v`
#### Populate the network descriptor file
- Create `custom-mnist/classifier.yaml` and insert:
```
---
# HWC (little data) configuration for parking
# MNIST classifier Model
# Compatible with MAX78002

arch: ai87net-mnist-classifier
dataset: mnist_2828

layers:
  # Layer 0: step1
  - in_offset: 0xA000
    out_offset: 0x0000
    processors: 0x0000000000000001
    output_processors: 0x00000000ffffffff
    operation: conv2d
    kernel_size: 3x3
    pad: 1
    activate: ReLU
  # Layer 1: step2
  - out_offset: 0xA000
    processors: 0x00000000ffffffff
    output_processors: 0xffffffffffffffff
    operation: conv2d
    kernel_size: 3x3
    pad: 1
    max_pool: 2
    pool_stride: 2
    activate: ReLU
  # Layer 2: step3
  - out_offset: 0x0000
    processors: 0xffffffffffffffff
    output_processors: 0x0000ffffffffffff
    operation: conv2d
    kernel_size: 3x3
    pad: 1
    max_pool: 2
    pool_stride: 2
    activate: ReLU
  # Layer 3: pool3
  - out_offset: 0xA000
    processors: 0x0000ffffffffffff
    output_processors: 0x0000ffffffffffff
    operation: Passthrough
    max_pool: 2
    pool_stride: 2
    activate: None
  # Layer 4: fc1
  - out_offset: 0x0000
    processors: 0x0000ffffffffffff
    output_processors: 0xffffffffffffffff
    operation: mlp
    flatten: true
    activate: ReLU
  # Layer 5: fc2
  - out_offset: 0xA000
    processors: 0xffffffffffffffff
    output_processors: 0x00000007ffffffff
    operation: mlp
    activate: None
    output_width: 32

```

#### Synthesize the network, as you would normally do:
-  `python ai8xize.py --test-dir CNN_example --prefix ai87net-mnist-classifier --checkpoint-file custom-mnist/qat_class_mnist_checkpoint_q8.pth.tar --config-file custom-mnist/classifier.yaml --device MAX78002 --timer 0 --display-checkpoint --verbose --softmax`

Lastly, within the `CNN_example/ai87net-mnist-classifier` directory, you should find the generated CNN files.