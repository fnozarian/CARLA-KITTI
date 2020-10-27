## OpenPCDet Setup & Installation

This is a step-by-step guide for installing [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) in a *nix distro.

#### 1) OpenPCDet Prerequisites

**1.1) Installing CUDA and CuDNN**

As a very first step, make sure `CUDA 10.2` and `CuDNN7` are properly installed on your machine and the necessary `PATH`s are correctly set. If not, follow this guide [cuda-10.x installation on Ubuntu](https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0) for a fresh installation.

Just for the sake of completeness, I list all the commands here. Open a terminal and do the following:

```bash
# cleaning up previous installations
$ sudo rm /etc/apt/sources.list.d/cuda*
$ sudo apt-get remove --autoremove nvidia-cuda-toolkit
$ sudo apt-get remove --autoremove nvidia-*
$ sudo apt-get purge nvidia*
$ sudo apt-get autoremove
$ sudo apt-get autoclean
$ sudo rm -rf /usr/local/cuda*

# fresh installation
$ sudo apt-get update
$ sudo add-apt-repository ppa:graphics-drivers
$ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
$ sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
$ sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

# actual installation happens here
$ sudo apt-get update
$ sudo apt-get install cuda-10-2
$ sudo apt-get install libcudnn7

# set PATH
$ sudo vim ~/.profile

# add the following at the end of the file.
# set PATH for cuda 10.2 installation
if [ -d "/usr/local/cuda-10.2/bin/" ]; then
    export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```



Reboot your system and verify proper installation of the drivers using:

```bash
$ nvidia-smi    # nvidia drivers
$ nvcc -V       # cuda
$ /sbin/ldconfig -N -v $(sed ‘s/:/ /’ <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn       # cudnn
```



With that done, but before installing *OpenPCDet*, create and activate a virtual or conda environment. For instance,

```bash
$ conda create -n openpcdet python=3.8
$ conda activate openpcdet
```

Then, install the following packages:

```bash
$ conda install -c conda-forge numpy ipython
$ conda install -c pytorch pytorch=1.6.0 torchvision
```

**1.2 Installing spconv**

As a *mandatory* requirement, we need to install the [**spconv**](https://github.com/traveller59/spconv) package. Else OpenPCDet will fail.
Since **no** binaries are distributed in the [spconv Git repository](https://github.com/traveller59/spconv), following the Docker path is the easiest option since the Docker image comes with a pre-built `.egg` binary. Further, it is also easier to build a `.whl` binary if that's desired. Hence, make sure Docker is installed on your machine. Then, launch a terminal and do:

```bash
$ sudo docker pull scrin/dev-spconv
$ sudo docker run -ti scrin/dev-spconv /bin/bash
```

The first command pulls the `spconv` docker image from the [dockerhub](https://hub.docker.com/r/scrin/dev-spconv), following which the second command runs the image. Now, we can use all the command line functionalities. At this point, there are two options:

- building a `.whl` file yourself
- using the pre-built and distributed `.egg` file

The `spconv-1.2.1-py3.8-linux-x86_64.egg` file can be found in `spconv/dist`.

Open another terminal (on the local machine where you want your wheel or egg files), then do:

`$ sudo docker ps` to find out the docker container process ID. It should be some alphanumeric chars (e.g., `e4137315390e`)

`cd` to the folder where you want to place your `.egg` file. This can be any folder but make sure the environment where you want to install the spconv package is activated (e.g., `openpcdet`). Then, do:        # note the trailing dot in the command (i.e., current folder)

```bash
$ sudo docker cp <CONTAINER ID>:/root/spconv/dist/spconv-1.2.1-py3.8-linux-x86_64.egg .
```

where `<container ID>` is the spconv docker container ID.

This marks the end of using the pre-built `.egg` file of spconv for installation.

On the other hand, to build a `.whl` file, navigate to the terminal where the container is still running & then do:

```bash
$ python setup.py bdist_wheel
```

The above command will build the wheel, hopefully successfully, and write it to the `spconv/dist` folder. It'll be named something like the following: `spconv-1.2.1-cp38-cp38-linux_x86_64.whl`.

Now, to copy the wheel to a local host machine, follow same procedure as for the `.egg` file. That is,

- open a new terminal on the host machine
- find the container ID using `docker ps` (`$ sudo docker ps`)
- copy the wheel file to the destination folder as in:

```bash
$ sudo docker cp <CONTAINER ID>:/root/spconv/dist/spconv-1.2.1-cp38-cp38-linux_x86_64.whl .
```

Now, we can install spconv in the desired environment using either of the following:

```bash
$ easy_install spconv-1.2.1-py3.8-linux-x86_64.egg
$ pip install spconv-1.2.1-cp38-cp38-linux_x86_64.whl
```

#### 2) Installing OpenPCDet:

Activate the environment where you want to install OpenPCDet. For instance, following our example:

```bash
$ conda activate openpcdet
```

Install the packages found in `requirements.txt`.

Now we can finally install **OpenPCDet**. There are two ways to get the source code of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

1. Download the `tar.gz` or `.zip` release from [OpenPCDet/releases](https://github.com/open-mmlab/OpenPCDet/releases). (I'm using `v0.3.0`)
2. `$ git clone --recursive https://github.com/open-mmlab/OpenPCDet.git`

After that,  and then do:

```bash
$ cd OpenPCDet
$ python setup.py develop
```

The `develop` option will not actually install OpenPCDet but it will create symlinks in `site-packages` of your environment to the OpenPCDet directory on your machine. With this option, you commit to saying that the OpenPCDet path remains unchanged forever or you will not delete the folder. If you move the directory elsewhere, OpenPCDet will not work until you do the compilation again.

If everything works out fine, you wont encounter any issues. Now, launch an IPython terminal and check the installation of OpenPCDet using,

```python
In [1]: import pcdet

In [2]: pcdet.__version__
Out[2]: '0.3.0+0000000'
```



------------

Some helpful resources:

- [`spconv` installation guide](https://github.com/traveller59/spconv/blob/master/README.md#docker)
- [OpenPCDet installation guide](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)
- [`setup.py`](https://stackoverflow.com/a/39811884)

