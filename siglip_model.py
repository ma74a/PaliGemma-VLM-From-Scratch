import torch
import torch.nn as nn

from typing import Tuple

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
        
class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = (self.head_dim) ** -.05 # Equivalent to 1 / sqrt(self.head_dim)
        self.attention_dropout = config.attention_dropout
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_state shape -> [B, N, D]
        batch_size, num_patches, _ = hidden_state.shape
        
        # shape -> [B, N, D]
        query_states = self.q_proj(hidden_state)
        # shape -> [B, N, D]
        key_states = self.k_proj(hidden_state)
        # shape -> [B, N, D]
        value_states = self.v_proj(hidden_state)
        
        # split the embed_dim across each head by embed_dim // num_heads
        # shape after split -> [B, N, num_heads, head_dim]
        # then transpose num_heads and N so shape -> [B, num_heads, N, head_dim]
        query_states = query_states.veiw(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.veiw(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.veiw(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        
        # calculate the attention using Q*K^T / sqrt(d_k)
        # Q ->                 [B, num_heads, N, head_dim]
        # K after transpose -> [B, num_heads, head_dim, N]
        # shape -> [B, num_heads, N, N]
        atten_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        # check dims
        if atten_weights.size() != (batch_size, self.num_heads, num_patches, num_patches):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, num_patches, num_patches)}, but is"
                f" {atten_weights.size()}"
            )
        
        # applying softmax
        # shape -> [B, num_heads, N, N]
        atten_weights = nn.functional.softmax(atten_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        atten_weights = nn.functional.dropout(atten_weights, p=self.dropout, training=self.training)
        
        # atten_weights shape -> [B, num_head, N, N]
        # V shape -> [B, num_heads, N, head_dim]
        # output shape -> [B, num_heads, N, head_dim]
        atten_outputs = torch.matmul(atten_weights, value_states)
        # check dims
        if atten_outputs.size() != (batch_size, self.num_heads, num_patches, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, num_patches, self.head_dim)}, but is"
                f" {atten_outputs.size()}"
            )
            
        # before transpose -> [B, num_heads, N, head_dim]
        #  after transpose(2, 1) make it [B, N, num_heads, head_dim] -> swap N with num_heads
        # view(batch_size, num_tokens, num_heads*head_dim)
        atten_outputs = atten_outputs.transpose(2, 1).contiguous().view(batch_size, num_patches, self.num_heads*self.head_dim)
        
        # pass it to out proj
        # shape -> [B, N, D]
        atten_outputs = self.out_proj(atten_outputs)
        
        return atten_outputs

        
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
    