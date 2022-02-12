import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model

os.environ['CPP_TF_MIN_LOG_LEVEL'] = '2'



class MLP(layers.Layer):
    """
    Multilayer perceptron for Swin Transformer
    """
    def __init__(self, hidden_features=None, out_features=None, drop=0.):
        self.fc1 = layers.Dense(hidden_features)
        self.act = tfa.layers.GELU()
        self.fc2 = layers.Dense(out_features)
        self.drop = layers.Dropout(drop)


    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        if training:
            x = self.drop(x)
        return x



class SwinTransformerBlock(layers.Layer):
    """
    Swin Transformer Block.
    """
    def __init__(self,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.):
        """
        initialize function for SwinTransformerBlock
        :param input_resolution: tuple(int, int) input image resolution
        :param num_heads: int, number heads to be used in self attention
        :param window_size: int, window size in W-SMA
        :param shift_size: int, shift size in SW-MSA
        :param mlp_ratio: int, MLP ratio
        :param qkv_bias: boolean, use bias for QKV or not
        :param qk_scale: float, QKV scale
        :param drop: float, drop out rate for MLP layer
        :param attn_drop: float, attention drop out rate
        """
        super(SwinTransformerBlock, self).__init__()
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"



    def call(self, inputs, training=None):
        return inputs



class PatchEmbedding(layers.Layer):
    """
    Patch embedding layer
    """
    
    def __init__(self, img_size=224, patch_size=4, embed_dim=96):
        """
        initialization function
        :param img_size: int, image size default is 224
        :param patch_size: int, patch size default is 4
        :param embed_dim: number of linear projection output channels, default is 96
        """
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_dim = embed_dim
        assert img_size % patch_size == 0, f"img_size {img_size} should be divided by patch_size {patch_size}."

        self.H, self.W = img_size // patch_size, img_size // patch_size
        self.num_patches = self.H * self.W
        self.project = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        self.norm = layers.LayerNormalization()



    def call(self, inputs, *args, **kwargs):
        # (b, 224, 224, 3) -> (b, 56, 56, 96)
        x = self.project(inputs)
        #  (b, 56, 56, 96) -> (b, 3136, 96)
        x = tf.reshape(x, [-1, self.num_patches, self.embedding_dim])
        # (b, 3136, 96)
        x = self.norm(x)
        return x, (self.H, self.W)


class PatchMerging(layers.Layer):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, batch_size, dim, input_resolution):
        super().__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.H, self.W = self.input_resolution
        assert self.H % 2 == 0 and self.W % 2 == 0, f"x size ({H}*{W}) are not even."
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization()

    def forward(self, inputs):
        """
        x: B, H*W, C
        """
        x = tf.reshape(inputs, [self.batch_size, self.H, self.W, self.dim])
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = tf.reshape(x, [self.batch_size, self.H * self.W, 4 * self.dim])
        x = self.norm(x)
        x = self.reduction(x)
        return x




class BasicLayer(layers.Layer):
    """
    A basic Swin Transformer layer for one stage.
    """
    
    def __init__(self,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 downsample=None):
        """
        initialization function basic layer
        :param input_resolution: tuple, input resolution
        :param depth: int, number of blocks
        :param num_heads: int, number of attention heads
        :param window_size: int, local window size
        :param mlp_ratio: int, Ratio of mlp hidden dim to embedding dim.
        :param qkv_bias: (bool, optional) If True, add a learnable bias to query, key, value. Default: True
        :param qk_scale: (float | None, optional) Override default qk scale of head_dim ** -0.5 if set.
        :param drop: (float, optional) Dropout rate. Default: 0.0
        :param attn_drop: (float, optional) Attention dropout rate. Default: 0.0
        :param downsample: Downsample layer at the end of the layer. Default: None
        """
        super(BasicLayer, self).__init__()


        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        self.blocks = [SwinTransformerBlock(
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0 if (i % 2 == 0) else window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop) for i in range(depth)]

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution)
        else:
            self.downsample = None


    def call(self, inputs, *args, **kwargs):
        x = inputs
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)

        return x


class SwinTransformer(Model):
    """ Swin Transformer
        A Tensorflow impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 ape=False,
                 **kwargs):
        super(SwinTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.ape = ape
        self.weight_initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)

        # init patch embedding layer
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        # absolute position
        if self.ape:
            self.absolute_pos_embed = self.add_weight(shape=[1, self.patch_embed.num_patches, embed_dim], initializer=self.weight_initializer, name='absolute_pos_embed')

        self.pos_drop = layers.Dropout(drop_rate)


    def call(self, inputs, training=None, mask=None):
        x = self.forward_features(inputs)
        x = self.head(x)
        return x

    def forward_features(self, inputs):
        # (b, 224, 224, 3) -> (b, 56, 56, 96) -> (b, 3136, 96)
        x, (H, W) = self.patch_embed(inputs)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)


        return x


    def head(self, inputs):
        return inputs





if __name__ == '__main__':
    # forward testing
    fake_images = tf.random.normal([4, 224, 224, 3])
    # build model
    swinTransformer = SwinTransformer(ape=True)
    # forward
    outputs = swinTransformer(fake_images)
    # print outputs
    print(outputs.shape)

