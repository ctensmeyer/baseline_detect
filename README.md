# baseline_detect

This repo contains our code and models we submitted to the [ICDAR 2017 cBAD competition](https://scriptnet.iit.demokritos.gr/competitions/5/).  A short [write up]() and [presentation slides]() of our method are available.  A related baseline detection model can be found in the task2 folder of [this repo](https://github.com/ctensmeyer/HisDB).

This code depends on a number of python libraries: numpy, scipy, cv2 (python wrapper for opencv, v3), and caffe [(my custom fork)](https://github.com/ctensmeyer/caffe).

## Usage
The usage statement can be found by running `python detect_baselines.py`:
```
USAGE: python detect_baselines.py in_image out_txt [simple|complex] [gpu#] [weights]
	in_image is the input image to be labeled
	out_txt is the resulting baseline file
	gpu is an integer device ID to run networks on the specified GPU.  If omitted, CPU mode is used
```

The input is a document image (300 dpi), and you can specify the simple track model or the complex track model.  Each track has slightly different post processing as well.  gpu# is a device id and is typically 0.  If you want to supply your own caffe weights (for the model.caffemodel network arch), you can using the last argument.

## Docker

For those who don't want to install the dependencies, I have created a docker image to run this code. You must have the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin installed to use it though you can still run our models on CPU (not recommended).

An example command for the docker container is

```
nvidia-docker run -v $HOST_WORK_DIRECTORY:/data tensmeyerc/icdar2017:baseline_detect python detect_baselines.py /data/in_image.png /data/output_file.txt simple 0 
```

`$HOST_WORK_DIRECTORY` is a directory on your machine that is mounted on /data inside of the docker container (using -v).  It's the only way to expose images to the docker container.
There is no need to download the containers ahead of time.  If you have docker and nvidia-docker installed, running the above commands will pull the docker image (~2GB) if it has not been previously pulled.
