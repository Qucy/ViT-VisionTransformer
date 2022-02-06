import os
import tensorflow as tf
from PVT.pvtv1 import pvt_tiny
from PVT.pvtv2 import pvt_v2_b0, pvt_v2_b2_li

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

# init hyper parameter
batch_size = 64
AUTO_TUNE = tf.data.AUTOTUNE
lr = 1e-5

"""
Here for simple, we use CIFAR100 image for test only
You can use your own dataset as well but remember to update image size and patch size accordingly
"""
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y


def load_data(is_test):


    # loading data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    if is_test:
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

    return ds_train, ds_test


def train(name, model, epochs, is_test=False):

    ds_train, ds_test = load_data(is_test)

    earlyStopCallBack = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(ds_train, validation_data=ds_test, callbacks=[earlyStopCallBack], epochs=epochs)

    model.save(f'./model/{name}.tf')




## construct pvt v1 tiny
# pvt1_tiny = pvt_tiny(img_size=32, batch_size=batch_size, num_classes=100)
## train model
# train(name='pvt1-tiny', model=pvt1_tiny, epochs=1)

# construct pvt v2 b0
# pvt2_b0 = pvt_v2_b0(patch_size=3, img_size=32, num_classes=100)
# # train model
# train(name='pvt_v2_b0', model=pvt2_b0, epochs=1, is_test=True)

# construct pvt v2 b0
pvt_v2_b2_li = pvt_v2_b2_li(patch_size=3, img_size=32, num_classes=100)
# train model
train(name='pvt_v2_b2_li', model=pvt_v2_b2_li, epochs=5, is_test=False)



