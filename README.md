# Shelly Vision

> This project was built for [first tech challenge](https://www.firstinspires.org/robotics/ftc) but can be used for any object detection problem

Shelly vision is an agglomeration of files which allows the creation of custom object detection datasets that can be trained on locally with gpu.

## Some notes for FTC folks

- An already existing solution is to use the [ftc-ml](https://storage.googleapis.com/ftc-ml-firstinspires-prod/docs/ftc-ml_manual_2021.pdf) machine learning toolkit

- The machine learning toolkit is great for beginners but is annoying in that you only have a specific amount of training time, training is slow on external hardware, and only coaches have any real access to uploading and such. Plus, even creating datasets with the tool is hit or miss as it creates some strange errors in training on your own hardware.

- It might not make much sense to use this project if you don't have decent hardware, just use the ftc-ml tool (if that's an option for you)

> Note: ensure that in your `build.dependencies.gradle` has `implementation 'org.tensorflow:tensorflow-lite-task-vision:0.3.1'` under `dependencies {` otherwise it will fail

## Setup

> This will not work on all systems, there's always some weird error, I wish you the best of luck

- Install [Conda](https://www.anaconda.com/products/individual) and set it up using their instructions

- Create a new virtualenv using `conda create --name tf`

- Install tensorflow gpu (or just regular tensorflow with no gpu) using `conda install tensorflow-gpu`

- Install tflite model maker using `pip install tflite_model_maker`

- There's probably ones I missed, install those too

## Usage

- Upload your own input video like the one provided

- Change the .pbtxt to custom name or add if you'd like

- Run `data_collector.py` and watch it create lots of training data for you :D (after drawing first box)

- Run `generate_tfrecord.py` and create a record for tensorflow to train on (using the syntax in the top of file)

- Run `train.py` and watch as it (maybe) trains your custom object detection model!

A lot of the info on this is in the files, they're pretty short, I encourage you to look through them

This requires only minimal configuration and tweaking, I'm sure you'll get it!

![xkcd.com](https://imgs.xkcd.com/comics/will_it_work.png)

We'll improve this eventually
