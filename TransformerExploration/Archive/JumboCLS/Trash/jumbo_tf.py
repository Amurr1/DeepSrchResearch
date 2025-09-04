import tensorflow as tf
from tensorflow.keras import layers, activations, initializers
from typing import Union, Tuple, List
from vision_transformer_tf import VisionTransformer, LayerScale, DropPath, MLP, Block, resample_with_jcls_token

class JumboBlock(Block):
    """Extends Block with J "jumbo" CLS tokens and an extra MLP branch."""
    
    def __init__(
        self,
        J: int,
        jumbo_mlp_ratio: float,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        init_values: float,
        drop_path: float,
        mlp_dropout: float,
        **kwargs,
    ):
        # initialize standard transformer Block
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            drop_path=drop_path,
            mlp_dropout=mlp_dropout,
            **kwargs,
        )
        self.J = J
        self.dim = dim
        self.jumbo_dim = J * dim
        # additional LayerNorm + MLP + LayerScale + DropPath for jumbo tokens
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.jumbo_mlp = MLP(
            in_features=self.jumbo_dim,
            hidden_features=int(self.jumbo_dim * jumbo_mlp_ratio),
            drop=mlp_dropout,
        )
        self.ls3 = LayerScale(self.jumbo_dim, init_values=init_values)
        self.drop_path3 = DropPath(drop_path)

    def call(self,
             x: tf.Tensor,
             layer_idx: int,
             training: bool = False,
             return_after_attn: bool = False) -> tf.Tensor:
        # 1) attention branch (same as Block)
        x = x + self.drop_path1(
            self.ls1(self.attn(self.norm1(x))),
            training=training
        )
        if return_after_attn:
            return x

        # 2) process first J (jumbo) tokens
        x_cls = x[:, :self.J, :] # (B, J, dim)
        b = tf.shape(x_cls)[0]
        x_cls_flat = tf.reshape(x_cls, [b, self.J * self.dim]) # (B, J*dim)
        x_cls_flat = x_cls_flat + self.drop_path3(
            self.ls3(self.jumbo_mlp(self.norm3(x_cls_flat))),
            training=training
        )

        # 4) otherwise continue with patch tokens branch
        x_patches = x[:, self.J:, :]                      # (B, N, dim)
        x_patches = x_patches + self.drop_path2(
            self.ls2(self.mlp(self.norm2(x_patches), training=training)),
            training=training
        )

        # 5) recombine jumbo and patch tokens
        x_cls_reshaped = tf.reshape(x_cls_flat, [b, self.J, self.dim])  # (B, J, dim)
        x = tf.concat([x_cls_reshaped, x_patches], axis=1)
        return x


class Jumbo(VisionTransformer):
    """Vision Transformer with Jumbo CLS-token extension."""
    
    def __init__(
        self,
        J: int,
        jumbo_mlp_ratio: float,
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
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            mlp_dropout=mlp_dropout,
            init_values=init_values,
            **kwargs,
        )
        # number of CLS tokens
        self.J = J
        # override the class token weight to hold J tokens
        self.cls_token = self.add_weight(
            "cls_token",
            shape=(1, J, embed_dim),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        # rebuild the transformer blocks as JumboBlock
        self.blocks: List[JumboBlock] = []
        for i in range(depth):
            blk = JumboBlock(
                J,
                jumbo_mlp_ratio,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                drop_path=drop_path_rate,
                mlp_dropout=mlp_dropout,
                name=f"jumbo_block_{i}",
            )
            self.blocks.append(blk)
        # override normalization and head for flattened jumbo CLS output
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.head = layers.Dense(
            num_classes,
            kernel_initializer=initializers.Zeros(),
            bias_initializer=initializers.Constant(
                -6.9 if num_classes == 1000 else -9.25
            ),
        )

    def set_pos_embed(self, grid_size: Union[int, Tuple[int, int]]):
        # allow int or tuple
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        # resize, preserving J prefix tokens
        new_pos = resample_with_jcls_token(
            self.pos_embed,
            new_size=grid_size,
            old_size=self.grid_size,
            jumbo_size=self.J,
        )
        # assign and update grid size
        self.pos_embed.assign(new_pos)
        self.grid_size = grid_size

    def call(
        self,
        x: tf.Tensor,
        mask_ratio: float = None,
        training: bool = False,
    ) -> tf.Tensor:
        # 1) embed patches + add pos embeddings
        x = self.patch_embed(x)
        x = x + self.pos_embed
        # 2) optional random patch masking
        if mask_ratio is not None:
            b = tf.shape(x)[0]
            n = tf.shape(x)[1]
            num_masked = tf.cast(mask_ratio * tf.cast(n, tf.float32), tf.int32)
            rand = tf.random.uniform((b, n))
            rnd_idx = tf.argsort(rand, axis=-1)
            keep = rnd_idx[:, num_masked:]
            x = tf.gather(x, keep, batch_dims=1)
        # 3) prepend J class tokens
        cls_tokens = tf.broadcast_to(self.cls_token, [tf.shape(x)[0], self.J, self.embed_dim])
        x = tf.concat([cls_tokens, x], axis=1)
        # 4) pass through JumboBlocks
        for i, blk in enumerate(self.blocks):
            x = blk(x, layer_idx=i, training=training)
        # 5) **now** extract and flatten the first J tokens with a static reshape
        x_cls = x[:, :self.J, :] # (B, J, C)
        b = tf.shape(x_cls)[0]
        x_cls = tf.reshape(x_cls, [b, self.J * self.embed_dim]) # (B, J*C)
        # 6) normalize and head on final x (flattened at last block)
        x = self.norm(x)
        logits = self.head(x)
        return logits
