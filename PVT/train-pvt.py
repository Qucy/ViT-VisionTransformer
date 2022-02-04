import os
import tensorflow as tf
from PVT.pvt import pvt_tiny

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
checkpoint_filepath = './model/pvt-v1.h5'

# loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# for testing purpose - to quick validate whether all the function is workable
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# create datasets
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds_train = ds_train.cache().map(preprocess).shuffle(50000).batch(batch_size, drop_remainder=True).prefetch(buffer_size=AUTO_TUNE)

ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.cache().batch(batch_size, drop_remainder=True).map(preprocess)

# construct model
pvt_tiny = pvt_tiny(img_size=32, batch_size=batch_size, num_classes=100)

pvt_tiny.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

# callback for early stop
earlyStopCallBack = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# train
pvt_tiny.fit(ds_train, validation_data=ds_test, callbacks=[earlyStopCallBack], epochs=1)

# save model
pvt_tiny.save('./model/pvt.tf')

test_data, test_label = next(iter(ds_test))

print(test_data.shape)

prediction = pvt_tiny.predict(test_data)

print(prediction.shape)