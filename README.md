# Bachelor-Thesis

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
