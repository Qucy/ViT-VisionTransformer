import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)


class MLP(layers.Layer):

    def __init__(self, name, hidden_features, out_features, drop_rate=0):
        super(MLP, self).__init__()
        self.fc1 = layers.Dense(hidden_features, name=f'{name}_MLP_DENSE1')
        self.act = tfa.layers.GELU()
        self.fc2 = layers.Dense(out_features, name=f'{name}_MLP_DENSE2')
        self.drop = layers.Dropout(drop_rate, name=f'{name}_MLP_DROP')


    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        if training:
            x = self.drop(x)
        return x


class Attention(layers.Layer):

    def __init__(self, name, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads # dimension for each head
        self.scale = qk_scale or self.head_dim ** -.5

        self.q = layers.Dense(dim, use_bias=qkv_bias, name=f'{name}_Q')
        self.kv = layers.Dense(2 * dim, use_bias=qkv_bias, name=f'{name}_KV')
        self.attention_drop = layers.Dropout(attn_drop, name=f'{name}_ATTEN_DROP')
        self.proj = layers.Dense(dim, name=f'{name}_PROJ_DENSE')
        self.proj_drop = layers.Dropout(proj_drop, name=f'{name}_PROJ_DROP')

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = layers.Conv2D(dim, kernel_size=sr_ratio, strides=sr_ratio, name=f'{name}_SR_CONV2D')
            self.norm = layers.LayerNormalization()


    def call(self, inputs, H, W):
        # (b, 196, 768) -> (b, 196, num_heads, head_dim)
        q = layers.Reshape((-1, self.num_heads, self.head_dim))(self.q(inputs))
        # (b, 196, num_heads, head_dim)-> (b, num_heads, 196, head_dim)
        q = tf.transpose(q, [0, 2, 1, 3])

        if self.sr_ratio > 1:
            # (b, 196, 768) -> (b, 14, 14, 768)
            x_ = tf.reshape(inputs, [-1, H, W, self.dim])
            # (b, 14, 14, 768) -> (b, 7, 7, 768)
            x_ = self.sr(x_)
            # (b, 7, 7, 768) -> (b, 49, 768)
            x_ = tf.reshape(x_, [-1, H//self.sr_ratio * W//self.sr_ratio, self.dim])
            # (b, 49, 768)
            x_ = self.norm(x_)
            # (b, 49, 768) -> # (b, 49, 768 *2)
            kv = self.kv(x_)
            # (b, 49, 768 *2) -> (b, 49, 2, num_heads, head_dim)
            kv = tf.reshape(kv, [-1, H//self.sr_ratio * W//self.sr_ratio, 2, self.num_heads, self.head_dim])
            # (b, 49, 2, num_heads, head_dim) -> (2, b, num_heads, 49, head_dim)
            kv = tf.transpose(kv, [2, 0, 3, 1, 4])
        else:
            # (b, 196, 768) -> # (b, 49, 768 *2)
            kv = self.kv(inputs)
            # (b, 196, 768 *2) -> (b, 196, 2, num_heads, head_dim)
            kv = layers.Reshape((-1, 2, self.num_heads, self.head_dim))(kv)
            # (b, 49, 2, num_heads, head_dim) -> (2, b, num_heads, 49, head_dim)
            kv = tf.transpose(kv, [2, 0, 3, 1, 4])

        k, v = kv[0], kv[1]

        # calc attention
        # (b, num_heads, 196, 96) * (b, num_heads, 96, 196) ->  (b, num_heads, 196, 196)
        attention = tf.matmul(q, tf.transpose(k, [0,1,3,2])) * self.scale
        # (b, num_heads, 196, 196)
        attention = tf.nn.softmax(attention, axis=-1)
        attention = self.attention_drop(attention)
        # (b, num_heads, 196, 196) * # (b, num_heads, 196, 96) -> (b, num_heads, 196, 96)
        x = tf.matmul(attention, v)
        # (b, num_heads, 196, 96) -> (b, 196, num_heads, 96)
        x = tf.transpose(x, [0, 2, 1, 3])
        # (b, 196, num_heads, 96) -> (b, 196, num_heads * 96)
        x = layers.Reshape((-1, self.num_heads * self.head_dim))(x)
        # (b, N, C) -> (b, N, proj_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(layers.Layer):
    
    def __init__(self, name, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Block, self).__init__()
        self.norm1 = layers.LayerNormalization()
        self.attention = Attention(name, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio)
        self.norm2 = layers.LayerNormalization()
        self.mlp = MLP(name, hidden_features=int(dim * mlp_ratio), out_features=dim, drop_rate=drop)

    def call(self, inputs, H, W):
        x = self.norm1(inputs)
        x = self.attention(x, H, W)
        x = inputs + x

        y = self.norm2(x)
        y = self.mlp(y)

        return x + y


class PatchEmbedding(layers.Layer):

    def __init__(self, name, img_size=224, patch_size=16, embedding_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        assert img_size % patch_size == 0, f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.H, self.W = img_size // patch_size, img_size // patch_size
        self.num_patches = self.H * self.W
        self.project = layers.Conv2D(embedding_dim, kernel_size=patch_size, strides=patch_size, name=f'{name}_PATCH_CONV2D')
        self.norm = layers.LayerNormalization()


    def call(self, inputs):
        # (b, 224, 224, 3) -> (b, 14, 14, 768)
        x = self.project(inputs)
        #  (b, 14, 14, 768) -> (b, 196, 768)
        x = tf.reshape(x, [-1, self.num_patches, self.embedding_dim])
        # (b, 196, 768)
        x = self.norm(x)
        return x, (self.H, self.W)



class PyramidVisionTransformer(Model):

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 batch_size=64,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1]):
        """
        initialize function for PVC
        :param img_size: image size default value 224
        :param patch_size: patch size default value 4
        :param num_classes: number of classes default value 1000
        :param embed_dims: embedding dimension from stage 1 to 4
        :param num_heads: number of attention heads from stage 1 to 4
        :param mlp_ratios: MLP layer hidden layer ratio from stage 1 to 4
        :param qkv_bias: qkv using bias or not
        :param qk_scale:  qkv scale or not
        :param drop_rate: drop rate for MLP layer
        :param attn_drop_rate: attention drop rate
        :param depths: number of attention modules from stage 1 to 4
        :param sr_ratios: spatial reduction attention ratios from stage 1 to 4
        """
        super(PyramidVisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.batch_size = batch_size
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = PatchEmbedding(name='patch_embed1', img_size=img_size, patch_size=patch_size, embedding_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbedding(name='patch_embed2', img_size=img_size // 4, patch_size=2, embedding_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbedding(name='patch_embed3', img_size=img_size // 8, patch_size=2, embedding_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbedding(name='patch_embed4', img_size=img_size // 16, patch_size=2, embedding_dim=embed_dims[3])

        # position embedding
        self.weight_initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)
        self.pos_embed1 = None
        self.pos_embed2 = None
        self.pos_embed3 = None
        self.pos_embed4 = None
        self.pos_drop1 = layers.Dropout(drop_rate)
        self.pos_drop2 = layers.Dropout(drop_rate)
        self.pos_drop3 = layers.Dropout(drop_rate)
        self.pos_drop4 = layers.Dropout(drop_rate)

        # Blocks
        self.block1 = [Block(name=f'BLOCK1_{i}', dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, sr_ratio=sr_ratios[0]) for i in range(depths[0])]

        self.block2 = [Block(name=f'BLOCK2_{i}', dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, sr_ratio=sr_ratios[1]) for i in range(depths[1])]

        self.block3 = [Block(name=f'BLOCK3_{i}', dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, sr_ratio=sr_ratios[2]) for i in range(depths[2])]

        self.block4 = [Block(name=f'BLOCK4_{i}', dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, sr_ratio=sr_ratios[3]) for i in range(depths[3])]

        self.norm = layers.LayerNormalization()

        # class token
        self.cls_token = None

        # dense layer for prediction
        self.dense = layers.Dense(num_classes, name='CLASSIFICATION_DENSE')


    def get_config(self):
        config = {
            'weight_initializer' : self.weight_initializer
        }
        base_config = super(PyramidVisionTransformer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def build(self, input_shape):
        self.pos_embed1 = self.add_weight(shape=[1, self.patch_embed1.num_patches, self.embed_dims[0]], initializer=self.weight_initializer, name='pos_embed1')
        self.pos_embed2 = self.add_weight(shape=[1, self.patch_embed2.num_patches, self.embed_dims[1]], initializer=self.weight_initializer, name='pos_embed2')
        self.pos_embed3 = self.add_weight(shape=[1, self.patch_embed3.num_patches, self.embed_dims[2]], initializer=self.weight_initializer, name='pos_embed3')
        self.pos_embed4 = self.add_weight(shape=[1, self.patch_embed4.num_patches + 1, self.embed_dims[3]], initializer=self.weight_initializer, name='pos_embed4')
        self.cls_token = self.add_weight(shape=[1, 1, self.embed_dims[3]], initializer=self.weight_initializer, name='cls_token')
        super(PyramidVisionTransformer, self).build(input_shape)



    def call(self, inputs):

        # stage 1, patch image (b, 224, 224, 3) -> (b, 56, 56, 64) -> (b, 3136, 64)
        x, (H, W) = self.patch_embed1(inputs)
        # adding position embedding
        x = x + self.pos_embed1
        # dropout
        x = self.pos_drop1(x)
        # transformer encoder
        for blk in self.block1:
            x = blk(x, H, W) # (b, 3136, 64)

        # (b, 3136, 64) -> (b, 56, 56, 64)
        x = tf.reshape(x, [-1, H, W, self.embed_dims[0]])

        # stage 2 patch image (b, 56, 56, 64) -> (b, 28, 28, 64) -> (b, 784, 128)
        x, (H, W) = self.patch_embed2(x)
        # adding position embedding
        x = x + self.pos_embed2
        # dropout
        x = self.pos_drop2(x)
        # transformer encoder
        for blk in self.block2:
            x = blk(x, H, W) # (b, 784, 128)

        # (b, 784, 128) -> (b, 28, 28, 128)
        x = tf.reshape(x, [-1, H, W, self.embed_dims[1]])

        # stage 3 patch image (b, 28, 28, 128) -> (b, 14, 14, 256) -> (b, 196, 256)
        x, (H, W) = self.patch_embed3(x)
        # adding position embedding
        x = x + self.pos_embed3
        # dropout
        x = self.pos_drop3(x)
        # transformer encoder
        for blk in self.block3:
            x = blk(x, H, W) # (b, 196, 256)

        # (b, 196, 256) -> (b, 14, 14, 256)
        x = tf.reshape(x, [-1, H, W, self.embed_dims[2]])

        # stage 4 patch image (b, 14, 14, 256) -> (b, 7, 7, 512) -> (b, 49, 512)
        x, (H, W) = self.patch_embed4(x)
        # concat with cls token (b, 49, 512) -> (b, 50, 512)
        cls_token = tf.broadcast_to(self.cls_token, [self.batch_size, 1, self.embed_dims[3]])
        x = layers.concatenate([cls_token, x], axis=1)
        # adding position embedding
        x = x + self.pos_embed4
        # dropout
        x = self.pos_drop4(x)
        # transformer encoder
        for blk in self.block4:
            x = blk(x, H, W) # (b, 50, 512)

        # layer normalization
        x = self.norm(x)

        # extract cls token
        cls_token = x[:, 0]

        # dense layer to make prediction
        predictions = self.dense(cls_token)

        return predictions



def pvt_tiny(**kwargs):
    return PyramidVisionTransformer(patch_size=4,
                                    embed_dims=[64, 128, 320, 512],
                                    num_heads=[1, 2, 5, 8],
                                    mlp_ratios=[8, 8, 4, 4],
                                    qkv_bias=True,
                                    depths=[2, 2, 2, 2],
                                    sr_ratios=[8, 4, 2, 1],
                                    **kwargs)



def pvt_small(**kwargs):
    return PyramidVisionTransformer(patch_size=4,
                                    embed_dims=[64, 128, 320, 512],
                                    num_heads=[1, 2, 5, 8],
                                    mlp_ratios=[8, 8, 4, 4],
                                    qkv_bias=True,
                                    depths=[3, 4, 6, 3],
                                    sr_ratios=[8, 4, 2, 1],
                                    **kwargs)



def pvt_medium(**kwargs):
    return PyramidVisionTransformer(patch_size=4,
                                    embed_dims=[64, 128, 320, 512],
                                    num_heads=[1, 2, 5, 8],
                                    mlp_ratios=[8, 8, 4, 4],
                                    qkv_bias=True,
                                    depths=[3, 4, 18, 3],
                                    sr_ratios=[8, 4, 2, 1],
                                    **kwargs)



def pvt_large(**kwargs):
    return PyramidVisionTransformer(patch_size=4,
                                    embed_dims=[64, 128, 320, 512],
                                    num_heads=[1, 2, 5, 8],
                                    mlp_ratios=[8, 8, 4, 4],
                                    qkv_bias=True,
                                    depths=[3, 8, 27, 3],
                                    sr_ratios=[8, 4, 2, 1],
                                    **kwargs)


def pvt_huge_v2(**kwargs):
    return PyramidVisionTransformer(patch_size=4,
                                    embed_dims=[128, 256, 512, 768],
                                    num_heads=[2, 4, 8, 12],
                                    mlp_ratios=[8, 8, 4, 4],
                                    qkv_bias=True,
                                    depths=[3, 10, 60, 3],
                                    sr_ratios=[8, 4, 2, 1],
                                    **kwargs)



if __name__ == '__main__':

    inputs = tf.random.normal([4, 32, 32, 3])

    pvt = pvt_tiny(batch_size=4, img_size=32, num_classes=10)

    outputs = pvt(inputs)

    print(outputs.shape)
