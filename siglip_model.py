

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size: int=768,  # -> embedding_dim
        intermediate_size: int=3072, # -> ff_d
        num_hidden_layers: int=12, # -> number of transformer encoder blocks
        num_attention_heads: int=12, # -> number of attention heads
        num_channels: int=3,
        image_size: int=224,
        patch_size: int=16,
        layer_norm_eps: float=1e-6,
        attention_dropout: float=0.0,
        num_image_tokens: int = None
    ):
        self.hidden_size = hidden_size
        self.iintermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens