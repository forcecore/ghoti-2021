# Identification of Cichlid Fishes from Lake Malawi Using Deep Learning

This repository is a renewal for the research named
"Identification of Cichlid Fishes from Lake Malawi Using Computer Vision",
by Deokjin Joo, Ye-seul Kwan, Jongwoo Song, Catarina Pinho, Jody Hey and Yong-Jin Won.
(https://doi.org/10.1371/journal.pone.0077686)

This project introduces recent deep learning methods for
more accuracy with less human inputs and feature engineering.
Here, we apply only some of the most basic techniques as a tutorial for other researchers.

* The code for our old work is located here: https://github.com/forcecore/ghoti
* The photo (raw) of cichlids is available here: https://zenodo.org/record/5560501

## Environment Setup

This project is tested on Arch Linux, as of 2021-10,
with Python 3.9 (3.7 on Docker) + PyTorch 1.9.1.
Here, we assume that you are familiar with the Python programming language.
We recommend you setup the environment on a Linux machine since on Windows,
WSL2's GPU support and Docker's GPU support are still experimental, as of 2021.
Should you work on Windows, we recommend using Anaconda (https://www.anaconda.com).
With Anaconda's Python, PyTorch should work on Windows with GPU acceleration.

Practically, we recommend that you separate the deep learning machine from your working machine,
as the deep learning workload is likely to slow your machine down,
which in turn will slow your IDE and/or other paper writing programs.

### Virtual Environment Setup

For Python, it is a custom to setup a virtual environment for each project
to avoid *dependency hell*.

To create a new one, use the following commands:

```
$ python --version
Python 3.9.6

$ mkdir -p ~/usr
$ python -m venv --system-site-packages ~/usr/venv-tf39
$ source ~/usr/venv-tf39/bin/activate
```

You need to activate your environment every time you launch a new *shell*,
unless you add the activate command into *.bashrc* or alike.

### Installing the Requirements

Now we can install PyTorch and other needed Python libraries.
For PyTorch installation, please visit https://pytorch.org and follow their guid.

Currently, this project need these libraries other than PyTorch:

```
$ pip install dvc omegaconf pytorch_lightning sklearn
```

Now, let us test if GPU acceleration is available:

```
$ python
Python 3.9.7 (default, Aug 31 2021, 13:28:12)
[GCC 11.1.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>>
```

Press Ctrl+D to exit the Python interpreter shell.
If you see "True" then the GPU acceleration is available.
If you get "False", you need to install the GPU driver and the CUDA framework,
please search the internet on how, as we consider that to be beyond the scope of this document.

## Using Docker to Get the Environment

If you wish to use docker,
change directory into `./docker` then `./1_build.sh` to build the docker image.
After that, `./2_bash.sh` will give you the proper environment run this project.

## Running the Project

As we are classifying an image, we used an image classification tutorial from PyTorch Tutorials.
Also, as we used PyTorch Lightning framework,
which offers higher-level facilities to reduce the burden of programming.
The results are the codes in "ghoti" directory.

We used DVC (https://dvc.org) to control the data flow, as defined in `dvc.yaml`.
Typing `dvc repro` command should give you a train/test data split, training and evaluation run.

`./20_run_exps.sh` will give you 10 randomized trials.
After the trial runs are complete, `dvc exp show` will display the results.
The hyperparameters are defined in `params.yaml`.

### Trouble Shooting

```
  File "/home/jdj/work/ghoti-2021/ghoti/model.py", line 9, in __init__
    self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

urllib.error.HTTPError: HTTP Error 403: rate limit exceeded
```

Loading the pretrained weight for the ResNet gave us the above error.
You may or may not run into this error, depending on the PyTorch version.
As of 2021-09, for Torch 1.9.1, the error occurs and the walk-around is
to load the pretrained models like the following:

```
model = torchvision.models.densenet121(pretrained=True, progress=True)
```

## Experimental Results

We acquired the following results.
However, since we didn't fix the random seed for neural network initialization,
the results may vary if you re-run the random trials yourself.

| Trial no. | Accuracy | F1-score |
|-----------|----------|----------|
| 1         | 0.76271  | 0.7609   |
| 2         | 0.84746  | 0.8596   |
| 3         | 0.86441  | 0.6936   |
| 4         | 0.88136  | 0.77163  |
| 5         | 0.77966  | 0.76917  |
| 6         | 0.84746  | 0.80512  |
| 7         | 0.76271  | 0.6706   |
| 8         | 0.84746  | 0.82617  |
| 9         | 0.83051  | 0.73485  |
| 10        | 0.83051  | 0.81271  |

The average is an accuracy of 0.8254 ± 0.0423 and F1-score of 0.770435 ± 0.059 for the 10 trials above.
In our 2013 work, the accuracies were 0.6649 and 0.7562 for 48-feature-classifier and 82-feature-classifier, respectively.
This demonstrates the superiority of the deep learning methods.
We expect further improvement if more deep learning techniques are applied.

## Remark

We used a pretrained ResNet-50 neural network which is trained on ImageNet dataset.
However, there's a space for exploration.
You may use bigger networks or change the network architectures such as SqueezeNet, VGGNet, etc.

Also note that ImageNet does NOT represent real-world inputs in the wild,
hence switching to other pretrained networks that are trained on different datasets
may give you better results.
We redirect the readers to the following articles on this topic:

* https://crazyoscarchang.github.io/2019/02/16/seven-myths-in-machine-learning-research/#myth-2
* https://arxiv.org/abs/1902.06789
