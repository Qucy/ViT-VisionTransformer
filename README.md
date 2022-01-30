# ViT - Vision Transformer

### 1. Introduction

The paper is named 'An Image is Worth 16x16 words: Transformers for Image Recognition as Scale'  and published on ICLR2021 by Google Brain team. 

##### 1.1 Abstraction

- Transformer has become the de-facto standard for natural language processing tasks, but not widely used in computer vision
- In computer vision attention is is more like a supplement or enhancement
- ViT achieves a good performance on image classification tasks
- After trained on large datasets ViT can achieve a comparable performance with SOTA CNN

##### 1.2 Research background

Transformer achieved a huge successful in NLP but still not widely used in computer vision, this paper focus on whether a transformer can be applied in CV successfully. And transformer has below 3 major advantages:

- parallel computation
- global attention
- easily stack on top of each other

##### 1.3 Research result

ViT achieve a comparable performance with ResNet, below image depicting accuracy on different dataset.

![vit_results](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/vit_results.jpg)

- datasets: ImageNet, CIFAR, Oxford and Google private datasets JFT(20~30 time compared with ImageNet)
- ViT-H/14: ViT-Huge model, input sequence is 14x14

##### 1.4 Research meaning

- show that pure Transformer can be used in Computer Vision and achieve a good performance
- opening the Transformer research in Computer Vision



### 2. ViT architecture 

ViT overall architecture is depicting as below, it following the original Transformer and only using encoder. For input image it will split to N x N small patches and flatten the patches, add with position embedding then input the Transformer Encoder. Finally it will use zero position embedding to make classification prediction.

![vit](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/vit.jpg)



##### 2.1 Patch + Position Embedding

Patch + Position Embedding is mainly want to split the image into small patches and then give a position number to each patches to keep the sequence information. But this will lose the space information, like the patch 1 image is on top of patch 4 image.

![pos_embedding](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/pos_embedding.jpg)

So how to we split one image into small patches ? Here in ViT we use conv layers with specified strides then we can split one image into small patches. For example if we use a 16x16 kernel with 16x16 strides, so the cone layers will extract feature maps every 16 pixels and no overlap between each feature maps. If our input image is 224 x 224 x 3, with a 16 x 16 x 768 filters we can extract a 14 x 14 x 768 feature map.

![conv](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/conv.gif)

After we split one image into small patches, then we going flatten width and height, for example if our feature map is 14 x 14 x 768 then we going to resize it to 196 x 768.  After this we going to create a 1x768 CLS token which is **0*** in the below image. Then we going to concat our flattened feature maps and CLS token to generate a 197 x 768's feature map. After adding the CLS token we need to add position information into this feature map. We generate a 197 x 768's trainable matrix and add with feature map. 

![pos_embedding](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/pos_embedding.jpg)

Source code is as below

```python
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
```



##### 2.2 Transformer encoder

After we generate inputs with position information then we going to feed these inputs into the transformer encoder. Before we talked about transformer encoder, let's take a look at self attention first. To understand self attention, basically you just need to understand below images. In self attention, there are 3 major elements, Q stands for query vector, K stands for key vector and V stands for value vector.

![self_attention](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/self_attention.jpg)

To generate an output basically have below steps:

- First generate Q, K and V. Q = inputs * Wq,  K = inputs * Wk, V = inputs * Wv
- Then we use Q * K  to retrieve scores for each inputs
- Apply softmax function to scores to retrieve importance for each sequence
- Multiply these scores with V and calculate sum to generate outputs Z

![qkv](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/qkv.jpg)

A sample code is looks like below

```python
import numpy as np

def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.expand_dims(np.sum(t, axis=1), 1)
    return a

Query = np.array([
    [1,0,2],
    [2,2,2],
    [2,1,3]
])

Key = np.array([
    [0,1,1],
    [4,4,0],
    [2,3,1]
])

Value = np.array([
    [1,2,3],
    [2,8,0],
    [2,6,3]
])

scores = Query @ Key.T
print(scores)
scores = soft_max(scores)
print(scores)
out = scores @ Value
print(out)
```

##### 2.3 Multi-Head attention

Multi-Head attention is basically you have multiple Q, K and V. Below image depicting the 2 heads attention, here in the image we got 2 pairs of Q, K and V, hence we are going to get 2 attention outputs and then we can merge these outputs as final outputs.

![multi-head](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/multi-head.jpg)

source code for multi-head attention is as below

```python
class MultiHeadSelfAttention(layers.Layer):
    """
    Attention layer, split inputs into q,k,v vector
    then calc outputs based on formula outputs = ( Q * transpose(K) ) / sqrt(d) * V
    """
    def __init__(self, num_features, num_heads, dropout=.2):
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
```

##### 2.4 Transformer encoder

After Multi-Head attention is constructed, then we can start to constructed transformer encoder, below image depicting the network structure for transformer encoder.

- Embedding + Position -> Layer Normalization(yellow) -> Multi-Head Attention(green)
- Multi-Head Attention + (Embedding + Position) -> residual outputs 1
- residual outputs 1 -> Layer Normalization(yellow) -> MLP network(blue) 
- MLP network outputs +  residual outputs 1 -> final outpus



![transformer_encoder](https://github.com/Qucy/ViT-VisionTransformer/blob/master/img/transformer_encoder.jpg)

Source code for transformer encoder is as below

```python
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
```









 
