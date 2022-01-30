import os
import tensorflow as tf
from model import VisionTransformer

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'


"""
Here for simple, we use CIFAR100 image for test
"""

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# init hyper parameter
batch_size = 64
AUTO_TUNE = tf.data.AUTOTUNE
lr = 1e-5

# loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# create datasets
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.cache().map(preprocess).shuffle(50000).batch(batch_size, drop_remainder=True).prefetch(buffer_size=AUTO_TUNE)

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.cache().batch(batch_size, drop_remainder=True).map(preprocess)

# construct model
ViT = VisionTransformer(batch_size=batch_size, input_shape=[32, 32], patch_size=4, num_layers=6, num_classes=100)

ViT.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# callback for early stop
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# train
ViT.fit(ds_train, validation_data=ds_test, callbacks=[callback], epochs=5)
# save weights
ViT.save_weights('./model/ViT.h5')
