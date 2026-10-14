import torch
import torch.nn as nn

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
        
        
        
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.emebeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values shape -> [B, C, H, W]
        # after embeddings -> [B, num_patches, embed_dim]
        hidden_state = self.emebeddings(pixel_values)
        
        # after encoder -> [B, num_patches, embed_dim]
        last_hidden_state = self.encoder(hidden_state)
        
        # after layernorm -> [B, num_patches, embed_dim]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        
        return last_hidden_state
        
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values shape -> [B, C, H, W]
        # after vision_model -> [B, num_patches, embed_dim]
        return self.vision_model(pixel_values)