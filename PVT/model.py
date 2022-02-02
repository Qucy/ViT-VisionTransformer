import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential


os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)