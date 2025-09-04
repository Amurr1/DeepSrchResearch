from transformers import TFDeiTForImageClassification, DeiTFeatureExtractor
import tensorflow as tf

class JumboDeiT(TFDeiTForImageClassification):
    """
    Extends TFDeiTForImageClassification by adding J "jumbo" CLS tokens
    and an additional MLP head over those tokens.

    Usage:
        model = JumboDeiT.from_pretrained(
            "facebook/deit-base-patch16-224",   # model id
            from_pt=True,                       # load PyTorch weights
            J=4,                                # number of extra CLS tokens
            jumbo_mlp_ratio=2.0                 # expansion ratio for jumbo-MLP
        )
    """
    def __init__(
        self,
        config,                # HuggingFace config object for DeiT
        J=4,                   # number of extra CLS tokens
        jumbo_mlp_ratio=2.0,   # hidden dim multiplier for jumbo MLP
        **kwargs,
    ):
        # 1) Initialize parent TFDeiT model (loads model architecture)
        super().__init__(config, **kwargs)
        
        # 2) Store parameters
        self.J = J
        self.jumbo_mlp_ratio = jumbo_mlp_ratio
        hidden_size = config.hidden_size  # transformer hidden dimension

        # 3) Create learnable extra CLS token embeddings of shape (1, J, D)
        self.jumbo_cls = self.add_weight(
            name="jumbo_cls",
            shape=(1, J, hidden_size),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
        
        # 4) Build an MLP to process the flattened J CLS tokens
        #    It maps (B, J*D) -> (B, hidden_size) via LayerNorm, Dense, Dense
        self.jumbo_head = tf.keras.Sequential([
            # Normalize the flattened tokens
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            # Expand dimension: from D*J -> D*J*ratio
            tf.keras.layers.Dense(
                hidden_size * jumbo_mlp_ratio,
                activation="gelu",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
            ),
            # Project back down to D
            tf.keras.layers.Dense(
                hidden_size,
                kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
            ),
        ], name="jumbo_mlp")

    def call(self, pixel_values, **kwargs):
        """
        Forward pass:
        1) Run the base DeiT encoder to get (B, 1+N, D) embeddings.
        2) Prepend the J extra CLS tokens, replacing the original CLS.
        3) Extract and flatten the first J tokens, apply jumbo_head MLP.
        4) Re-insert the processed jumbo token into the sequence.
        5) Run the standard classification head on the modified sequence.
        """
        # --- 1) Base DeiT encoder ---
        # `outputs` includes last_hidden_state of shape (B, 1+N, D)
        outputs = super().deit(pixel_values, **kwargs)
        seq = outputs.last_hidden_state  # sequence tokens

        # --- 2) Prepend extra J tokens ---
        batch_size = tf.shape(seq)[0]
        # tile jumbo_cls to match batch: (1,J,D) -> (B,J,D)
        jumbo_cls = tf.broadcast_to(
            self.jumbo_cls,
            [batch_size, self.J, self.config.hidden_size]
        )
        # drop original CLS (seq[:,0,:]) and prepend jumbo_cls
        seq = tf.concat([jumbo_cls, seq[:, 1:, :]], axis=1)

        # --- 3) Process jumbo tokens separately ---
        # extract first J tokens: shape (B, J, D)
        cls_tokens = seq[:, : self.J, :]
        # flatten to (B, J*D)
        flat = tf.reshape(
            cls_tokens,
            [batch_size, self.J * self.config.hidden_size]
        )
        # run through jumbo_head: output shape (B, D)
        jumbo_out = self.jumbo_head(flat)

        # --- 4) Reassemble sequence ---
        # turn (B, D) -> (B,1,D)
        jumbo_out = tf.expand_dims(jumbo_out, axis=1)
        # concat processed jumbo token + remaining tokens
        seq = tf.concat([jumbo_out, seq[:, self.J :, :]], axis=1)

        # --- 5) Standard classification head ---
        # pooler: takes first token as [CLS], classifier: final Dense
        pooled = self.pooler(seq)            # (B, D)
        logits = self.classifier(pooled)     # (B, num_labels)
        return logits

# ---------------- Example Usage ----------------
# feature_extractor = DeiTFeatureExtractor.from_pretrained(
#     "facebook/deit-base-patch16-224"
# )
# model = JumboDeiT.from_pretrained(
#     "facebook/deit-base-patch16-224", from_pt=True,
#     J=4, jumbo_mlp_ratio=2.0
# )
# inputs = feature_extractor(images=images, return_tensors="tf")
# logits = model(inputs.pixel_values)