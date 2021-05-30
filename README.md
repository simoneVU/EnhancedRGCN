# Bachelor-Thesis

## Installation
### Conda Installation
1) The general installation instruction for Conda can be found[here](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html).
  1a) For Windows users, the specific instructions can be found [here](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/windows.html)
  1b) For Mac OS users, the specific instructions can be found [here](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/windows.html)
    1b')If you are on Mac and using the zsh with iTerm2,then, follow [this guide](https://medium.com/@sumitmenon/how-to-get-anaconda-to-work-with-oh-my-zsh-on-mac-os-x-7c1c7247d896) if the jupyter _conda_ command is not found.
  1c)For Linux users, the specific instructions can be found [here](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/linux.html), or, the following script can be run:
  ~~~~
  'wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh\
    -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda')
   ~~~~
  1d) After the installation of _conda_, you should verify that it has been intalled correctly by running the command 'conda list' in your terminal. If it runs, it should return a list of all packages installed:
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
### PyTorch Geometric Installation
2) The general installation instruction for pytorch_geometric can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).
    2a)As first, at least PyTorch 1.4.0 has to be installed. If it is not installed, there are different option for different users:
        1) If you are a Windows user, this [guide](https://medium.com/@bryant.kou/how-to-install-pytorch-on-windows-step-by-step-cc4d004adb2a) should lead you to a successful pytorch installation with conda
        2) If you are a Linux user,
