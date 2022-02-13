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


def window_partition(x, h, w, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    num_windows_h = h//window_size
    num_windows_w = w//window_size
    x = tf.reshape(x, [1, num_windows_h, window_size, num_windows_w, window_size, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [num_windows_w*num_windows_h, window_size, window_size, -1])
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, -1])
    return x


class WindowAttention(layers.Layer):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """
    def __init__(self, dim, batch_size, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        :param dim: (int): Number of input channels.
        :param batch_size: (int): batch size
        :param window_size: (tuple[int]): The height and width of the window.
        :param num_heads: (int): Number of attention heads.
        :param qkv_bias: (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        :param qk_scale: (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        :param attn_drop: (float, optional): Dropout ratio of attention weight. Default: 0.0
        :param proj_drop: (float, optional): Dropout ratio of output. Default: 0.0
        """
        super().__init__()
        self.dim = dim
        self.batch_size = batch_size
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.weight_initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=.02)

        # define a parameter table of relative position bias, shape -> 2*Wh-1 * 2*Ww-1, num_heads
        self.relative_position_bias_table = self.add_weight(shape=[1, (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads], initializer=self.weight_initializer)

        # get pair-wise relative position index for each token inside the window
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        # (W), (H) -> (2, W, H)
        coords = tf.stack(tf.meshgrid(coords_h, coords_w), axis=0)
        # (2, W, H) -> (2, W*H)
        coords_flatten = tf.reshape(coords, [2, -1])
        # (2, W*H) -> (2, W*H, W*H)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # (2, W*H, W*H) -> (W*H, W*H, 2)
        relative_coords = tf.transpose(relative_coords, [1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords = relative_coords.numpy()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # (W*H, W*H, 2) -> (W*H, W*H)
        self.relative_position_index = tf.reduce_sum(relative_coords, axis=-1)

        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)


    def call(self, inputs, mask):
        """
        :param inputs: input features with shape of (num_windows*B, N, C)
        :param mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        :return: forward result
        """
        # (B, N, C) -> (B, N, 3*C)
        qkv = self.qkv(inputs)
        # (B, N, 3*C) -> (B, N, 3, num_heads, C//self.num_heads)
        qkv = tf.reshape(qkv, [self.batch_size, -1, 3, self.num_heads, self.dim // self.num_heads])
        # (B, N, 3, num_heads, C//self.num_heads) -> (3, B, num_heads, N, C//self.num_heads)
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        # q,k,v -> (B, num_heads, N, C//self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # (B, num_heads, N, C//self.num_heads) * (B, num_heads, C//self.num_heads, N) -> (B, num_heads, N, N)
        attn = tf.matmul(q, tf.transpose(k, [0,1,3,2])) * self.scale

        # retrieve relative position bias (W^4)
        relative_position_bias = tf.gather(self.relative_position_bias_table, axis=0, indices=tf.reshape(self.relative_position_index, -1))
        # (W^2, W^2, num_heads)
        relative_position_bias = tf.reshape(relative_position_bias, [self.window_size ** 2, self.window_size ** 2, -1])
        # (num_heads, W^2, W^2)
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        # (1, num_heads, W^2, W^2)
        relative_position_bias = tf.expand_dims(relative_position_bias, axis=0)

        attn = attn + relative_position_bias

        if mask is not None:
            #nW = mask.shape[0]
            #attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            #attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # (B, num_heads, N, N) * (B, num_heads, N, C//self.num_heads) -> (B, num_heads, N, C//self.num_heads)
        x = tf.matmul(attn, v)
        # (B, num_heads, N, C//self.num_heads) -> (B, N, num_heads, C//self.num_heads)
        x = tf.transpose(x, [0, 2, 1, 3])
        # (B, N, num_heads, C//self.num_heads) -> (B, N, C)
        x = tf.reshape(x, [self.batch_size, -1, self.dim])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(layers.Layer):
    """
    Swin Transformer Block.
    """
    def __init__(self,
                 dim,
                 batch_size,
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
        :param dim, int, size of input channel
        :param batch_size, int, batch size
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
        self.dim = dim
        self.batch_size = batch_size
        self.input_resolution = input_resolution
        self.H = input_resolution[0]
        self.W = input_resolution[1]
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

        self.norm1 = layers.LayerNormalization()
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = layers.LayerNormalization()
        self.mlp = MLP(hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = tf.zeros((1, self.H, self.W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.H, self.W, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, float(-100.0), float(0.0))
        else:
            attn_mask = None


    def call(self, inputs, training=None):
        shortcut = inputs
        x = self.norm1(inputs)
        x = tf.reshape(x, [self.batch_size, self.H, self.W, self.dim])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.H, self.W, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = tf.transpose(x_windows, [-1, self.window_size * self.window_size, self.dim]) # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, self.dim])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x

        x = tf.reshape(x, [self.batch_size, self.H * self.W, -1])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



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
    """ Patch Merging Layer.
    """
    def __init__(self, batch_size, dim, input_resolution):
        """
        :param batch_size: int, batch size
        :param dim: int, dim of input channel
        :param input_resolution: int, input resolution
        """
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
                 dim,
                 batch_size,
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
        :param dim: int, dim of input channel
        :param batch_size: int, batch size
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

        self.dim = dim
        self.batch_size = batch_size
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
            self.downsample = downsample(batch_size, dim, input_resolution)
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
                 ape=False):
        """
        :param img_size: (int): Input image size. Default 224
        :param patch_size: (int): Patch size. Default: 4
        :param num_classes: (int): Number of classes for classification head. Default: 1000
        :param embed_dim: (int): Patch embedding dimension. Default: 96
        :param depths: (tuple(int)): Depth of each Swin Transformer layer.
        :param num_heads: (tuple(int)): Number of attention heads in different layers.
        :param window_size: (int): Window size. Default: 7
        :param mlp_ratio: (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        :param qkv_bias: (bool): If True, add a learnable bias to query, key, value. Default: True
        :param qk_scale: (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        :param drop_rate: (float): Dropout rate. Default: 0
        :param attn_drop_rate: (float): Attention dropout rate. Default: 0
        :param ape: (bool): If True, add absolute position embedding to the patch embedding. Default: False
        """
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
    # # forward testing
    # fake_images = tf.random.normal([4, 224, 224, 3])
    # # build model
    # swinTransformer = SwinTransformer(ape=True)
    # # forward
    # outputs = swinTransformer(fake_images)
    # # print outputs
    # print(outputs.shape)
    import numpy as np

    H, W = 8, 8
    window_size = 2
    shift_size = 1

    # calculate attention mask for SW-MSA
    img_mask = np.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, H, W, window_size)  # nW, window_size, window_size, 1

    mask_windows = tf.reshape(mask_windows, [-1, window_size*window_size])
    attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
    attn_mask = tf.where(attn_mask!=0, float(-100.0), float(0.0))
    print(attn_mask)






