import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential


os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)


class ClassToken(layers.Layer):
    """
    Class Token is the feature map that will be used in the last step for classification
    If the input image size is (224,224) and convolution filter is 16*16 with a strides=16, we will get feature map (14, 14)
    After we flatten this feature map to 196 which means this image have 196 sequence features
    Then we going to create a class token and stack on this 196 features makes it 197
    During the training, this class token will interact with other feature maps and in the end we going to use this class token to make prediction
    """

    def __init__(self, initializer='zeros', regularizer=None, constraint=None, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)
        self.num_features = 0
        self.cls_w = None


    def build(self, input_shape):
        self.num_features = input_shape[-1]
        self.cls_w = self.add_weight(
            shape = (1, 1, self.num_features),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint
        )

    # def get_config(self):
    #     pass


    def call(self, inputs, *args, **kwargs):
        batch_size = inputs.shape[0]
        cls_broadcast = tf.broadcast_to(self.cls_w, [batch_size, 1, self.num_features])
        cls_broadcast = tf.cast(cls_broadcast, dtype=inputs.dtype)
        return tf.concat([cls_broadcast, inputs], axis=1)



class PositionEmbedding(layers.Layer):
    """
    Position Embedding layer: add position information to each small patches
    For instance, if input image is 224x224, after flatten it, it will 196x768 after concat with cls token it will be 197x764
    Hence position embedding layer will be 197x764 as well to provide position information for each feature
    """
    def __init__(self, initializer='zero', regularizer=None, constraint=None, **kwargs):
        super(PositionEmbedding, self).__init__()
        self.initializer = keras.initializers.get(initializer)
        self.regularizer = keras.regularizers.get(regularizer)
        self.constraint = keras.constraints.get(constraint)


    def build(self, input_shape):
        assert (len(input_shape) == 3), f"Input share should be 3 dimension, but got {len(input_shape)}"
        self.pos_w = self.add_weight(
            shape = (1, input_shape[1], input_shape[2]),
            initializer=self.initializer,
            regularizer=self.regularizer,
            constraint=self.constraint
        )


    def call(self, inputs, *args, **kwargs):
        return inputs + tf.cast(self.pos_w, dtype=inputs.dtype)



class MultiHeadSelfAttention(layers.Layer):
    """
    Attention layer, split inputs into q,k,v vector
    then calc outputs based on formula outputs = ( Q * transpose(K) ) / sqrt(d) * V
    """
    def __init__(self, num_features, num_heads, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_features = num_features
        self.num_heads = num_heads
        self.project_dim = num_features // num_heads
        self.qkv = layers.Dense(3 * self.num_features)
        self.dense = layers.Dense(self.num_features)
        self.dropout = layers.Dropout(dropout)


    def call(self, inputs, training=None):
        """
        :param inputs: input feature map with shape (batch_size, sequence_length, 3 * num_features)
        :return: processed inputs
        """
        batch_size = inputs.shape[0]
        # (batch_size, sequence_length, num_features) => (batch_size, sequence_length, 3 * num_features)
        qkv = self.qkv(inputs)
        # (batch_size, sequence_length, 3 * num_features) -> (batch_size, sequence_length, 3, num_heads, project_dim)
        inputs = tf.reshape(qkv, [batch_size, -1, 3, self.num_heads, self.project_dim])
        # (batch_size, sequence_length, 3, num_heads, project_dim) -> (3, batch_size, num_heads, sequence_length, project_dim)
        inputs = tf.transpose(inputs, [2, 0, 3, 1, 4])
        # retrieve q,k,v -> shape (batch_size, num_heads, sequence_length, project_dim)
        query, key, value = inputs[0], inputs[1], inputs[2]
        # calculate score if sequence_length = 197 and project_dim = 64 (b, num_heads, 197, 64) @ (b, num_heads, 197, 64).T -> (b, num_heads, 197, 197)
        score = tf.matmul(query, key, transpose_b=True)
        # calculate scaled score
        scaled_score = score / tf.sqrt(tf.cast(self.project_dim, dtype=score.dtype))
        # calculate weights (b, num_heads, 197, 197)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        # calculate weighted value (b, num_heads, 197, 197) @ (b, num_heads, 197, 64) -> (b, num_heads, 197, 64)
        weighted_value = tf.matmul(weights, value)
        # (b, num_heads, 197, 64) -> (b, 197, num_heads, 64)
        outputs = tf.transpose(weighted_value, [0, 2, 1, 3])
        # (b, 197, num_heads, 64) -> (b, 197, num_heads*64)
        outputs = tf.reshape(outputs, [batch_size, -1, self.num_features])
        # (b, 197, num_heads*64) => (b, 197, num_heads*64)
        outputs = self.dense(outputs)
        if training:
            outputs = self.dropout(outputs)
        return outputs


class TransformerBlock(layers.Layer):

    def __init__(self, num_features, num_heads, mlp_dim, dropout=.2):
        super(TransformerBlock, self).__init__()
        self.layerNorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.multiHeadSelfAttention = MultiHeadSelfAttention(num_features, num_heads, dropout)
        self.dropout1 = layers.Dropout(dropout)
        self.layerNorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.MLP = Sequential([
            layers.Dense(mlp_dim),
            tfa.layers.GELU(),
            layers.Dropout(dropout),
            layers.Dense(num_features)
        ])
        self.dropout2 = layers.Dropout(dropout)


    def call(self, inputs, training=None):
        # layer normalization
        x = self.layerNorm1(inputs)
        # multi-head attention
        x = self.multiHeadSelfAttention(x)
        # dropout 1
        if training:
            x = self.dropout1(x)
        # residual
        x = layers.Add()([inputs, x])
        # layer normalization
        y = self.layerNorm2(x)
        # MLP
        y = self.MLP(y)
        # dropout 2
        if training:
            y = self.dropout2(y)
        # residual
        y = layers.Add()([x, y])

        return y





        

if __name__ == '__main__':
    # test ClassToken
    feature_maps = tf.random.normal([4, 196, 768])
    # init clsToken
    # clsToken = ClassToken()
    # inputs = clsToken(feature_maps)
    # assert inputs.shape == (4, 197, 768)
    #
    # # test PositionEmbedding
    # posEmbedding = PositionEmbedding()
    # inputs_posEmbedding = posEmbedding(inputs)
    # print(inputs_posEmbedding.shape)
    #
    # # test attention
    # attention = MultiHeadSelfAttention(num_features=768, num_heads=12)
    # attention_outputs = attention(inputs_posEmbedding)
    # print(attention_outputs.shape)

    transformer = TransformerBlock(num_features=768, num_heads=12, mlp_dim=3072)
    outputs = transformer(feature_maps)

    print(outputs.shape)