from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, activations, initializers
from typing import Tuple, Union, List

"""
Implementation of timm.layers.DropPath using tensorflow.keras.layers.
DropPath is a regularization technique that drops connections between neurons, not neurons themselves.
"""
class DropPath(layers.Layer):

    def __init__(self, drop_prob: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training: bool = None):
        if (not training) or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape = (batch, 1, 1) works for (B, N, C) tensors
        shape = (tf.shape(x)[0], 1, 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_mask = tf.floor(random_tensor)
        return x / keep_prob * binary_mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prob": self.drop_prob})
        return cfg

"""
Converts patches of the image into tokens
"""
class PatchEmbed(layers.Layer):

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        in_chans: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
            use_bias=True,
        )

    def call(self, x):
        # x: (B, H, W, C)
        x = self.proj(x)  # (B, H/ps, W/ps, embed_dim)
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [b, h * w, c])  # (B, N, C)
        return x

"""
MLP class that ensures alignment with paper's timm structure
"""
class MLP(layers.Layer):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(
            hidden_features,
            activation=activations.gelu,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
        )
        self.drop1 = layers.Dropout(drop)
        self.fc2 = layers.Dense(
            in_features, kernel_initializer=initializers.TruncatedNormal(stddev=0.02)
        )
        self.drop2 = layers.Dropout(drop)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x

"""
Damps incoming information but can be tuned during training to increase
the signal for more important sections of the previous layer.
"""
class LayerScale(layers.Layer):
    def __init__(self, dim: int, init_values: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.gamma = self.add_weight(
            "gamma",
            shape=(dim,),
            initializer=tf.keras.initializers.Constant(init_values),
            trainable=True,
        )

    def call(self, x):
        return x * self.gamma

"""
Attention class that ensures alignment with paper's timm structure
"""
class Attention(layers.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert dim % num_heads == 0 #dim should be divisible by num_heads
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=qkv_bias,
            kernel_initializer=initializers.TruncatedNormal(stddev=0.02),
        )
        self.proj = layers.Dense(
            dim, kernel_initializer=initializers.TruncatedNormal(stddev=0.02)
        )

    def call(self, x):
        b = tf.shape(x)[0] # batch size
        n = tf.shape(x)[1] # sequence length

        qkv = self.qkv(x)  # (B, N, 3*dim)
        qkv = tf.reshape(
            qkv, [b, n, 3, self.num_heads, self.head_dim]
        )  # (B, N, 3, heads, dh)
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # (3, B, heads, N, dh)
        q, k, v = tf.unstack(qkv, axis=0)

        # TODO: Research scaled dot product attention
        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)  # (B, heads, N, N)
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.matmul(attn, v)  # (B, heads, N, dh)

        embed_dim = self.num_heads * self.head_dim
        x = tf.transpose(x, [0, 2, 1, 3])  # (B, N, heads, dh)
        x = tf.reshape(x, [b, n, embed_dim])  # (B, N, dim)
        x = self.proj(x)
        return x

"""
Block class that ensures alignment with paper's timm structure
"""
class Block(layers.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float = 1e-4,
        drop_path: float = 0.1,
        mlp_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=mlp_dropout,
        )
        self.ls2 = LayerScale(dim, init_values=init_values)
        self.drop_path2 = DropPath(drop_path)

    def call(self, x, training=False):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))), training=training)
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), training=training)), training=training)
        return x


"""
ViT class that ensures alignment with paper's timm structure
"""
class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        mlp_dropout: float = 0.0,
        init_values: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Compute grid size
        if isinstance(img_size, int):
            self.grid_size = (img_size // patch_size, img_size // patch_size)
        else:
            self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        num_patches = self.grid_size[0] * self.grid_size[1]

        # Layers / weights
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            name="patch_embed",
        )
        self.pos_embed = self.add_weight(
            "pos_embed",
            shape=(1, num_patches, embed_dim),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        self.cls_token = self.add_weight(
            "cls_token",
            shape=(1, 1, embed_dim),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

        # Stochastic depth decay rule – fixed rate like original
        dpr = [drop_path_rate] * depth

        self.blocks: List[Block] = []
        for i in range(depth):
            blk = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                drop_path=dpr[i],
                mlp_dropout=mlp_dropout,
                name=f"block_{i}",
            )
            self.blocks.append(blk)

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(
            num_classes,
            kernel_initializer=initializers.Zeros(),
            bias_initializer=initializers.Constant(
                -6.9 if num_classes == 1000 else -9.25
            ),
        )

    # ---------------------------------------------------------------------

    def call(self, x, training=False):
        x = self.patch_embed(x)  # (B, N, C)
        x = x + self.pos_embed
        cls_tokens = tf.broadcast_to(self.cls_token, [tf.shape(x)[0], 1, self.embed_dim])
        x = tf.concat([cls_tokens, x], axis=1)  # (B, 1+N, C)

        for blk in self.blocks:
            x = blk(x, training=training)

        x = self.norm(x)
        cls = x[:, 0]  # (B, C)
        logits = self.head(cls)
        return logits

    # Convenience helpers --------------------------------------------------

    def turn_off_scaled_dot(self):
        for blk in self.blocks:
            blk.attn.turn_off_scaled_dot()



"""
Allows the transformer to interpolate to different image sizes by 
reducing the number of patches considered in the attention sequence
"""
def resample_with_jcls_token(
    posemb: tf.Tensor,
    new_size: Tuple[int, int],
    old_size: Tuple[int, int],
    jumbo_size: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> tf.Tensor:
    
    if new_size == old_size:
        return posemb

    jcls_tok, patch_posemb = (
        posemb[:, :jumbo_size], 
        posemb[:, jumbo_size:],
    )

    # Reshape patch part
    c = tf.shape(patch_posemb)[-1]
    patch_posemb = tf.reshape(patch_posemb, [1, old_size[0], old_size[1], c])
    patch_posemb = tf.image.resize(
        patch_posemb, size=new_size, method=interpolation, antialias=antialias
    )
    patch_posemb = tf.reshape(patch_posemb, [1, new_size[0] * new_size[1], c])

    # Recombine
    return tf.concat([jcls_tok, patch_posemb], axis=1)