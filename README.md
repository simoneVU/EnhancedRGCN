# Potential Energy to Improve Link Prediction with Relational Graph Neural Networks
<p align="center">
<a href="https://github.com/pytorch/pytorch"><img src="https://img.shields.io/badge/pyTorch-11.1-blue.svg"></a>
<a href="https://github.com/pyg-team/pytorch_geometric"><img src="https://img.shields.io/badge/pyorchGometric-1.7.2-orange.svg"></a>
<a href=" https://github.com/simoneVU/EnhancedRGCN/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
<a href=""><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
</p>
<p align="center">
<img width="674" alt="Screen Shot 2022-01-18 at 4 37 11 PM" src="https://user-images.githubusercontent.com/60779914/149968740-e19b7016-a42a-4dfa-91f8-24f26dd74062.png">
</p>

This repository contains the code used for the experiments in the paper.

```
Potential Energy to Improve Link Prediction with Relational Graph Neural Networks.
Simone Colombo, Dimitris Alivanistos and Michael Cochez
```

**Abstract**: Potential Energy (PE) between 2 bodies with mass, refers to the relative gravitational pull between them. Analogously, in the context of a graph, nodes can thought of as objects where a) the product of the degrees of nodes acts as a proxy for mass, b) the clustering coefficients of common neighbours as a proxy for gravitational acceleration, and c) the inverse of the shortest distance between nodes as a proxy for distance in space, which allows for PE calculation as introduced in prior work. In this work, we are investigating the effects of incorporating PE in Link Prediction (LP) with Relational Graph Convolutional Networks (R-GCN). Specifically, we explore the benefits of including PE calculation as an informative prior to the LP task and in a follow-up experiment as a learnable feature to predict. We performed several experiments and show that considering PE in the LP process has certain advantages and find that the information PE provides was not captured by the embeddings produced by the R-GCN.

## Model descriptions
**Vanilla R-GCN**: The R-GCN for LP with an additional Dense Layer to reduce the dimension of the textual embeddings

**eR-GCN**: The energized R-GCN - utilizing the additional signal of PE for prediction, note that PE comes directly from the input graph topology, also note the trainable parameter 𝛼 that represents signal weight. 

**PE Estimator**: An MLP-Regressor trained to predict the PE between nodes. 

**eR-GCN-MLP**: The combination of eR-GCN and PE estimator that optimizes 𝛼 based on both the R-GCN and the PE Estimator. The light blue box indicates frozen parameters, which allows us to isolate the effects of specific parts in the architecture.


## Installation
### Conda Installation
1) The general installation instruction for **Conda** can be found [here](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html).
    1. For **Windows users**, the specific instructions can be found [here](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/windows.html)
    2. For **Mac OS X users**, the specific instructions can be found [here](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/windows.html)
        * If you are on Mac and using the zsh with iTerm2,then, follow [this guide](https://medium.com/@sumitmenon/how-to-get-anaconda-to-work-with-oh-my-zsh-on-mac-os-x-7c1c7247d896) if the jupyter _conda_ command is not found.
        * Another option for MAC OS X user would be running the followinf script:
   ~~~~
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh\
    -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
   ~~~~
    3.**For Linux users**, the specific instructions can be found [here](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/linux.html), or, the following script can be run:
  ~~~~
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh\
    -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
   ~~~~
  4. For **Linux and Mac users only**, to run 'conda' from the terminal, run 'export PATH="$HOME/miniconda/bin:$PATH"'. If you want do not want to run it each time you start the system, add it to ~/.bashrc for Linux users or to ~/.zshrc for Mac OS X users. Then, run source ~/.bashrc (or ~/.zshrc) to update it.
  5. After the installation of _conda_, you should verify that it has been installed correctly by running the command 'conda list' in your terminal. If it runs, it should return a list of all packages installed:
 ~~~~
packages in environment at /opt/anaconda3/envs/pyg:

Name                    Version                   Build  Channel
ase                       3.21.1                   pypi_0    pypi
blas                      1.0                         mkl  
ca-certificates           2021.4.13            hecd8cb5_1  
cached-property           1.5.2                    pypi_0    pypi
cctools                   927.0.2              h5ba7a2e_4  
certifi                   2020.12.5        py36hecd8cb5_0  
chardet                   4.0.0                    pypi_0    pypi')
...                         ...                       ...
~~~~
6.Then, update conda and initialize the conda environment by running 
~~~~
conda init
conda update conda
conda create -n conda3.6 python=3.6
source activate conda3.6
~~~~
### PyTorch Geometric Installation
2) The general installation instruction for pytorch_geometric can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).
    1. As first step, at least **PyTorch 1.4.0** has to be installed. If it is not installed, there are different option for different users:
        * If you are a **Windows user**, this [guide](https://medium.com/@bryant.kou/how-to-install-pytorch-on-windows-step-by-step-cc4d004adb2a) should lead you to a successful pytorch installation with conda.
        * If you are a **Linux or Mac OS X user** you can run 'conda install pytorch torchvision -c pytorch' in the terminal to install it.
    2. As second step, you can follow the general instruction in the link at 2). 
3) **For Mac OS X** users only, it is possible to use the script pyg_setup.sh by running 'sh ./pyg_setup.sh' in the terminal and this will create the conda environment, activate it, install pytorch and install all the necessary dependencies for pytorch_geometric.

## Run the code
Once pytorch_geometric is installed, copy the pytorch_geometric folder from github to your own directory with 'git clone https://github.com/rusty1s/pytorch_geometric.git'. Then, do the same for the files in this repository with 'git clone https://github.com/simoneVU/Bachelor-Thesis.git'. Therefore, to run the code do '/opt/anaconda3/envs/pyg/bin/python /Users/simonecolombo/Desktop/VU_Amsterdam_CS_2020_2021/Triply:BachThesis/Bachelor-Thesis/link_prediction.py'.
