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
        
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        # this where we convert the image into series of patches
        # where each patches has its own embed dim
        self.patch_embbedings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"
        )
        self.flatten = nn.Flatten(start_dim=2)
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # print(self.num_patches) # 196
        # for each patch will be a position for it
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values shape -> [B, C, H, W]
        # after conv layer will be [B, embed_dim, H/patch_size, W/patch_size]
        patch_embedd = self.patch_embbedings(pixel_values)
        # print(patch_embedd.shape)
        
        # then flatten it to be [B, embed_dim, H/patch_size*W/patch_size]
        # then transpose last two dim
        # will be of shape -> [B, N, D]
        # B for batch_size
        # N for num of patches
        # D for embedd dim
        embeddings = self.flatten(patch_embedd).transpose(1, 2)
        # print(embeddings.shape)
        
        # then add position embeddings
        # shape will be as it is [B, N, D]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # print(embeddings.shape)
        
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
        
        
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
    
    
if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = SiglipVisionEmbeddings(SiglipVisionConfig())
    out = model(x)
    