import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

class MLP(layers.Layer):
    """
    MLP layer in PVT-v2 mainly added Depth Wise Conv layers after 1st FC layer
    """
    def __init__(self, name, hidden_features, out_features, drop_rate=0, linear=False):
        super(MLP, self).__init__()
        self.linear = linear
        self.hidden_features = hidden_features
        self.fc1 = layers.Dense(hidden_features, name=f'{name}_MLP_DENSE1')
        self.DConv = layers.DepthwiseConv2D(3, strides=(1, 1), padding='same')
        self.act = tfa.layers.GELU()
        self.fc2 = layers.Dense(out_features, name=f'{name}_MLP_DENSE2')
        self.drop1 = layers.Dropout(drop_rate, name=f'{name}_MLP_DROP1')
        self.drop2 = layers.Dropout(drop_rate, name=f'{name}_MLP_DROP2')


    def call(self, inputs, H, W, training=None):
        # [B, N, C] -> [B, N, hidden_features]
        x = self.fc1(inputs)
        if self.linear:
            x = tf.nn.relu(x)
        # [B, N, hidden_features] -> [B, H, W, hidden_features]
        x = tf.reshape(x, [-1, H, W, self.hidden_features])
        x = self.DConv(x)
        # [B, H, W, hidden_features] -> [B, N, hidden_features]
        x = tf.reshape(x, [-1, H*W, self.hidden_features])
        x = self.act(x)
        if training:
            x = self.drop1(x)
        x = self.fc2(x)
        if training:
            x = self.drop2(x)

        return x


class Attention(layers.Layer):
    """
    Attention module for PVT-V2, add averaging pooling when scaling KV if linear set to true
    """

    def __init__(self, name, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads # dimension for each head
        self.scale = qk_scale or self.head_dim ** -.5
        self.linear = linear

        self.q = layers.Dense(dim, use_bias=qkv_bias, name=f'{name}_Q')
        self.kv = layers.Dense(2 * dim, use_bias=qkv_bias, name=f'{name}_KV')
        self.attention_drop = layers.Dropout(attn_drop, name=f'{name}_ATTEN_DROP')
        self.proj = layers.Dense(dim, name=f'{name}_PROJ_DENSE')
        self.proj_drop = layers.Dropout(proj_drop, name=f'{name}_PROJ_DROP')

        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = layers.Conv2D(dim, kernel_size=sr_ratio, strides=sr_ratio, name=f'{name}_SR_CONV2D')
                self.norm = layers.LayerNormalization()
        else:
            self.pool = tfa.layers.AdaptiveAveragePooling2D(2) # in paper use 7, but since we are testing on small image, we use 2 here
            self.sr = layers.Conv2D(dim, kernel_size=1, strides=1)
            self.norm = layers.LayerNormalization()
            self.act = tfa.layers.GELU()


    def call(self, inputs, H, W):
        # (b, 196, 768) -> (b, 196, num_heads, head_dim)
        q = layers.Reshape((-1, self.num_heads, self.head_dim))(self.q(inputs))
        # (b, 196, num_heads, head_dim)-> (b, num_heads, 196, head_dim)
        q = tf.transpose(q, [0, 2, 1, 3])

        if not self.linear:
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
        else:
            # (b, 196, 768) -> (b, 14, 14, 768)
            x_ = tf.reshape(inputs, [-1, H, W, self.dim])
            # (b, 14, 14, 768) -> (b, 2, 2, 768)
            x_ = self.pool(x_)
            # (b, 2, 2, 768) -> (b, 2, 2, 768)
            x_ = self.sr(x_)
            # (b, 2, 2, 768) -> (b, 4, 768)
            x_ = tf.reshape(x_, [-1, 4, self.dim])
            # (b, 4, 768)
            x_ = self.norm(x_)
            # (b, 4, 768)
            x_ = self.act(x_)
            # (b, 4, 768) -> # (b, 4, 768 *2)
            kv = self.kv(x_)
            # (b, 4, 768 *2) -> (b, 4, 2, num_heads, head_dim)
            kv = tf.reshape(kv, [-1, 4, 2, self.num_heads, self.head_dim])
            # (b, 4, 2, num_heads, head_dim) -> (2, b, num_heads, 4, head_dim)
            kv = tf.transpose(kv, [2, 0, 3, 1, 4])

        # retrieve k,v
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
    
    def __init__(self, name, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super(Block, self).__init__()
        self.norm1 = layers.LayerNormalization()
        self.attention = Attention(name, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, sr_ratio, linear)
        self.norm2 = layers.LayerNormalization()
        self.mlp = MLP(name, hidden_features=int(dim * mlp_ratio), out_features=dim, drop_rate=drop)

    def call(self, inputs, H, W):
        x = self.norm1(inputs)
        x = self.attention(x, H, W)
        x = inputs + x

        y = self.norm2(x)
        y = self.mlp(y, H, W)

        return x + y


class OverlapPatchEmbed(layers.Layer):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        assert patch_size > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.H, self.W = img_size // stride, img_size // stride
        self.num_patches = self.H * self.W
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=stride, padding='same')
        self.norm = layers.LayerNormalization()


    def call(self, inputs):
        x = self.proj(inputs)
        x = tf.reshape(x, [-1, self.H*self.W, self.embed_dim])
        x = self.norm(x)
        return x, self.H, self.W



class PyramidVisionTransformerV2(Model):
    """
    Pyramid Vision Transformer V2
    Some default parameters have been changed due to i'm test on CIFAR100 dataset
    For 224,224 size image or other large image need to change below parameters
    img_size to your really image size
    num_classes to your category classes
    patch_size change patch size to 7 for stage 1 and to 4 for other stages
    """
    def __init__(self,
                 patch_size=4,
                 img_size=32,
                 num_classes=100,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 num_stages=4,
                 linear=False):
        """
        initialize function for PVC-v2
        :param img_size: image size default value 224
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
        :param num_stages: number of stages
        :param linear: apply linear layer or not
        """
        super(PyramidVisionTransformerV2, self).__init__()

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** i),
                                            patch_size=patch_size,
                                            stride=2,
                                            embed_dim=embed_dims[i])

            block = [Block(name=f'Block_{i}_{j}', dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop_rate, attn_drop=attn_drop_rate, sr_ratio=sr_ratios[i], linear=linear) for j in range(depths[i])]
            norm = layers.LayerNormalization()

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = layers.Dense(num_classes)



    def call(self, inputs):
        x = inputs
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = tf.reshape(x, [-1, H, W, self.embed_dims[i]])

        x = tf.reduce_mean(x, axis=1)

        return self.head(x)



def pvt_v2_b0(**kwargs):
    return PyramidVisionTransformerV2(embed_dims=[32, 64, 160, 256],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      depths=[2, 2, 2, 2],
                                      sr_ratios=[8, 4, 2, 1],
                                      **kwargs)


def pvt_v2_b1(**kwargs):
    return PyramidVisionTransformerV2(embed_dims=[64, 128, 320, 512],
                                       num_heads=[1, 2, 5, 8],
                                       mlp_ratios=[8, 8, 4, 4],
                                       qkv_bias=True,
                                       depths=[2, 2, 2, 2],
                                       sr_ratios=[8, 4, 2, 1],
                                       **kwargs)

def pvt_v2_b2(**kwargs):
    return PyramidVisionTransformerV2(embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      depths=[3, 4, 6, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      **kwargs)



def pvt_v2_b3(**kwargs):
    return PyramidVisionTransformerV2(embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      depths=[3, 4, 18, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      **kwargs)


def pvt_v2_b4(**kwargs):
    return PyramidVisionTransformerV2(embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      depths=[3, 8, 27, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      **kwargs)


def pvt_v2_b5(**kwargs):
    return PyramidVisionTransformerV2(embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[4, 4, 4, 4],
                                      qkv_bias=True,
                                      depths=[3, 6, 40, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      **kwargs)


def pvt_v2_b2_li(**kwargs):
    return PyramidVisionTransformerV2(embed_dims=[64, 128, 320, 512],
                                      num_heads=[1, 2, 5, 8],
                                      mlp_ratios=[8, 8, 4, 4],
                                      qkv_bias=True,
                                      depths=[3, 4, 6, 3],
                                      sr_ratios=[8, 4, 2, 1],
                                      linear=True,
                                      **kwargs)



if __name__ == '__main__':
    # test forward
    inputs = tf.random.normal([4, 32, 32, 3])
    pvtV2 = PyramidVisionTransformerV2(img_size=32, num_classes=100, linear=True)
    outputs = pvtV2(inputs)
    print(outputs.shape)



