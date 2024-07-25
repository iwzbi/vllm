from transformers import AutoConfig
import torch

config = AutoConfig.from_pretrained("/media/model-space/Llama-2-7b-hf")
"""
cache_engine.cpu_cache:  [layer1kvcache, layer2kvcache...]
layer1kvcache.shape=(2, num_blocks, block_size, num_kv_heads, head_size)
1 k block shape: (block_size, num_kv_heads, head_size)
"""
num_hidden_layers = config.num_hidden_layers
block_size = 16
num_kv_heads = config.num_key_value_heads
head_size = config.hidden_size // config.num_key_value_heads

# create 1 block tensor
shape = [num_hidden_layers, 2, block_size, num_kv_heads, head_size]
random_tensor = torch.rand(shape)
torch.save(random_tensor, "kvcache.pth")
