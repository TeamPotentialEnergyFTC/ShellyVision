import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import numpy as np
import os

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel(logging.DEBUG)

# older implementation
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9))
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)

# newer implementation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# refrence https://www.tensorflow.org/lite/tutorials/model_maker_object_detection#quickstart for the diffrences--the actual latency is not accurate with tflite_model_maker 
# (but if you really want to dive in deep) you can create your own raw tensorflow model, optimize it and convert it to tensorflow lite for similar numbers (I think)
spec = model_spec.get("efficientdet_lite1")

from utils import load_pbtxt, ANNOTATIONS_PATH, TFRECORD_PATH
labels = load_pbtxt(ANNOTATIONS_PATH + "labelmap.pbtxt")
label_map = {label_n + 1:labels[label_n] for label_n in range(len(labels))}
print(label_map)

# tfrecord pattern, size, label_map
train_ds = object_detector.DataLoader(TFRECORD_PATH + "train.record", len(os.listdir("data/images")), label_map) # or, for second argument, input a number
# optional but reccomended?
# test_ds = object_detector.DataLoader("./*/eval_dataset.record-*" , 4, label_map)

model = object_detector.create(train_ds, model_spec=spec, epochs=20, batch_size=8, do_train=True, train_whole_model=False)

# model.evaluate(test_data)
model.export(export_dir='.', tflite_filename="edge_model.tflite")
# model.evaluate_tflite("edge_model.tflite", test_ds)