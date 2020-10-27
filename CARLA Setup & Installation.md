## CARLA Setup & Installation

A step-by-step guide for installing CARLA.

#### 1) CARLA Prerequisites 

Before installing CARLA, we need to install the **PyGame** package.
However, before installing PyGame, we need to install the following packages at the (Linux) system level:

```bash
$ sudo apt-get install python3-pip python-dev libsdl-image1.2-dev
$ sudo apt-get install libsdl-mixer1.2 libsdl-mixer1.2-dev libsdl-ttf2.0 libsdl-ttf2.0-dev
$ sudo apt-get install libportmidi-dev libfreetype6-dev
```



#### 2) Installing Anaconda

Next we will have to install a Python distribution. Here, we will prefer the Anaconda distribution. 

```bash
$ wget -c https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
$ bash Anaconda3-2020.07-Linux-x86_64.sh
```



#### 3) Conda environment

For encapsulating the required packages in a self-contained environment, we will use a conda environment and install the Python packages required for running CARLA.

```bash
$ conda create -n carla python=3.8
$ conda activate carla
```

Now, we are ready to install PyGame in the conda env that we just created:

```bash
$ pip install -U pygame
```

We also need some additional but necessary packages:

```bash
$ conda install -c conda-forge numpy
```



#### 4) Installing CARLA

Now, we are ready to install CARLA from the pre-compiled binaries

- First, download a suitable (development) version from the below URL. 
- https://github.com/carla-simulator/carla/blob/master/Docs/download.md
  (I am using the vanilla version of [Carla 0.9.10](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.tar.gz)).  (Not RSS)

- Also, download the corresponding `AdditionalMaps` tar.gz file that comes bundled with a specific CARLA version.
  (I'm using [AdditionalMaps-0.9.10](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.10.tar.gz))

After downloading, extract the `tar.gz` files to the desired location. Mine is at `/home/mario/CARLA_0.9.10`.

#### 5) Running CARLA

First, to make the CARLA python distribution binary visible in `PYTHONPATH`, define the following in `.bashrc` or `.zshrc`:

```bash
CARLAHOME="/home/mario/CARLA_0.9.10"
CARLAPYTHONDIST="$CARLAHOME/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
export PYTHONPATH=$CARLAPYTHONDIST
```


Since we will also run CARLA to collect (training) data using [`carla-kitti-data-collector`](https://iceland.sb.dfki.de/bitbucket/users/farzad.nozarian/repos/carla-kitti-data-collector/browse), we need the following paths as well:

```bash
CARLAPYAPI="$CARLAHOME/PythonAPI"
export PYTHONPATH=$PYTHONPATH:$CARLAPYAPI:"$CARLAPYAPI/carla":"$CARLAPYAPI/carla/agents":"$CARLAPYAPI/examples"
```


Another alias needed by `carla-kitti-datacollector/data_collector.py`:

```bash
CARLA_ROOT=$CARLAHOME
export CARLA_ROOT
```



Compile the changes with:

```bash
$ source ~/.bashrc
```



Now, run the CARLA server:

```bash
$ cd ~/carla_0.9.10
$ bash ImportAssets.sh
$ bash CarlaUE4.sh
```

If everything works out fine, you will see a PyGame window showing buildings, road, etc.

Running Python API examples or scenario_runner:

```bash
$ conda activate carla
$ python spawn_npc.py
```

