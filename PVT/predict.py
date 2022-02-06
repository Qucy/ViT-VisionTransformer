import os
import tensorflow as tf

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

batch_size = 64

# loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)


ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.cache().batch(batch_size, drop_remainder=True).map(preprocess)

test_data, test_label = next(iter(ds_test))

print(test_data.shape)

model = tf.keras.models.load_model('./model/pvt_v2_b0.tf')

predictions = model.predict(test_data, batch_size=batch_size)

print(predictions.shape)
print(tf.math.softmax(predictions[0]))












