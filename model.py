import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


print(tf.__version__)



class ClassToken(layers.Layer):
    """
    Class Token is the feature map that will be used in the last step for classification
    If the image input size is (224,224) and convolution filter is 16*16 with a strides=16, we will get feature map (14, 14)
    After we flatten this feature map to 196 which means this image have 196 sequence features
    Then we going to create a class token and stack on this 196 features makes it 197
    During the training, this class token will interact with other feature maps and in the end we gong to use class token with dense layer to make prediction
    """

    def __init__(self, inputs, initializer='zeros', regularizer=None, constraint=None, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.num_features = 0
        self.cls_w = None


    def build(self, input_shape):
        self.num_features = input_shape[-1]
        self.cls_w = self.get_weights(
            shape = (1, 1, self.num_features),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint
        )

    def get_config(self):
        pass


    def call(self, inputs, *args, **kwargs):
        batch_size = inputs.shape[0]
        cls_broadcast = tf.broadcast_to(self.cls_w, [batch_size, 1, self.num_features])
        cls_broadcast = tf.cast(cls_broadcast, dtype=inputs.dtype)
        return tf.concat([cls_broadcast, inputs], axis=1)
