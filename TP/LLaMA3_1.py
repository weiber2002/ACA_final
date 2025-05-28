# Load model directly
import math
from typing import List, Optional, Tuple, Union
import os, torch, torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.activations import ACT2FN
from transformers.cache_utils import  Cache, DynamicCache, StaticCache
from transformers import LlamaConfig


import torch
import torch.nn as nn
import torch.distributed as dist

torch.manual_seed(100)
# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



# class TPAttention(nn.Module):
#     """
#     Megatron-style tensor-parallel attention that:
#     • row-shards Q
#     • *replicates* K,V when #kv_heads < tp_size (MQA/GQA)
#     • column-shards O
#     • gathers K,V before matmul, all-reduces after O-proj
#     """

#     def __init__(self, orig_attn: LlamaAttention, tp_size: int, group=None):
#         super().__init__()
#         assert orig_attn.config.hidden_size % tp_size == 0, "d_model divisible by TP"
#         self.group = group or dist.group.WORLD
#         self.rank  = dist.get_rank(self.group)
#         cfg        = orig_attn.config
#         self.d_k   = cfg.hidden_size // cfg.num_attention_heads

#         # ----- query projection: row-parallel -----
#         heads_per_rank   = cfg.num_attention_heads // tp_size
#         self.q_proj = nn.Linear(
#             cfg.hidden_size,
#             heads_per_rank * self.d_k,
#             bias=cfg.attention_bias,
#         )
#         q_slice = slice(self.rank * heads_per_rank * self.d_k,
#                         (self.rank+1)*heads_per_rank * self.d_k)
#         with torch.no_grad():
#             self.q_proj.weight.copy_(orig_attn.q_proj.weight[q_slice])
#             if cfg.attention_bias:
#                 self.q_proj.bias.copy_(orig_attn.q_proj.bias[q_slice])

#         # ----- key / value: replicate if too small -----
#         shard_kv = cfg.num_key_value_heads >= tp_size
#         kv_heads_per_rank = (cfg.num_key_value_heads // tp_size) if shard_kv else cfg.num_key_value_heads

#         self.k_proj = nn.Linear(cfg.hidden_size,
#                                 kv_heads_per_rank * self.d_k,
#                                 bias=cfg.attention_bias)
#         self.v_proj = nn.Linear(cfg.hidden_size,
#                                 kv_heads_per_rank * self.d_k,
#                                 bias=cfg.attention_bias)

#         with torch.no_grad():
#             if shard_kv:
#                 kv_slice = slice(self.rank * kv_heads_per_rank * self.d_k,
#                                  (self.rank+1)*kv_heads_per_rank * self.d_k)
#                 self.k_proj.weight.copy_(orig_attn.k_proj.weight[kv_slice])
#                 self.v_proj.weight.copy_(orig_attn.v_proj.weight[kv_slice])
#             else:        # replicate
#                 self.k_proj.weight.copy_(orig_attn.k_proj.weight)
#                 self.v_proj.weight.copy_(orig_attn.v_proj.weight)

#         # ----- output projection: column-parallel -----
#         self.o_proj = nn.Linear(
#             cfg.hidden_size,
#             cfg.hidden_size // tp_size,
#             bias=cfg.attention_bias,
#         )
#         col_slice = slice(self.rank * cfg.hidden_size // tp_size,
#                           (self.rank+1) * cfg.hidden_size // tp_size)
#         with torch.no_grad():
#             self.o_proj.weight.copy_(orig_attn.o_proj.weight[:, col_slice])
#             if cfg.attention_bias:
#                 self.o_proj.bias.copy_(orig_attn.o_proj.bias[col_slice])

#     def forward(self, hidden_states, *_, **__):
#         q = self.q_proj(hidden_states)                # [B, T, d/q_tp]
#         k = self.k_proj(hidden_states)                # replicate or shard
#         v = self.v_proj(hidden_states)

#         # --- gather K,V if they were sharded ---
#         if k.shape[-1] < self.k_proj.out_features * dist.get_world_size(self.group):
#             k = all_gather_last_dim(k, group=self.group)   # utility fn
#             v = all_gather_last_dim(v, group=self.group)

#         q = split_heads(q, self.d_k)                 # [B, h_tp, T, d_k]
#         k = split_heads(k, self.d_k)
#         v = split_heads(v, self.d_k)

#         attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
#         attn = attn.softmax(dim=-1)
#         out  = torch.matmul(attn, v)                 # [B, h_tp, T, d_k]
#         out  = merge_heads(out)                      # [B, T, d_model/tp]
#         out  = self.o_proj(out)                      # local slice
#         dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.group, async_op=True).wait()
#         return out
    


import copy
class Single_Device_LlamaDecoderLayer(nn.Module):
    def __init__(self, oigin_layer:LlamaDecoderLayer  , config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx= layer_idx
        self.self_attn_device0 = Single_Device_MyCustom_LlamaAttention(oigin_layer.self_attn , rank=0 , world_size=2    )
        self.self_attn_device1 = Single_Device_MyCustom_LlamaAttention(oigin_layer.self_attn , rank=1 , world_size=2   )
        # self.self_attn = copy.deepcopy( oigin_layer.self_attn )
        self.mlp_device0 = Single_Device_MyCustomMLP(oigin_layer.mlp , rank=0 , world_size=2    )
        self.mlp_device1 = Single_Device_MyCustomMLP(oigin_layer.mlp , rank=1 , world_size=2     )

        # self.mlp = copy.deepcopy( oigin_layer.mlp)
        # self.mlp = Single_Device_MyCustomMLP(config)
        self.input_layernorm = copy.deepcopy(oigin_layer.input_layernorm )
        self.post_attention_layernorm =copy.deepcopy(oigin_layer.post_attention_layernorm)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # print("decoder fowarding")
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        attn0, self_attn_weights0, wrong_present_key_value , k0 , v0 , cos0 , sin0= self.self_attn_device0(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        attn1, self_attn_weights1, wrong_present_key_value , k1 , v1 , cos1 , sin1= self.self_attn_device1(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        k_full = torch.cat([k0, k1], dim=1)
        v_full = torch.cat([v0, v1], dim=1)

        cos, sin = position_embeddings
        cache_kwargs = {"sin": sin0, "cos": cos0, "cache_position": cache_position}

        if use_cache:
            # print("use cache")
            if isinstance(past_key_value, Cache):           # Static/Dynamic cache case
                past_key_value.update(                      # in-place, returns None
                    k_full, v_full, self.layer_idx, cache_kwargs
                )
                present_key_value = past_key_value                  # must be None in this path
            else:   
                # print("legacy  cache")                                        # legacy tuple path
                present_key_value = (k_full, v_full)        # 2-tuple of tensors
        else:
            present_key_value = None

        # print("attn0 shape" , attn0)
        # print("attn1 shape" , attn1)
        hidden_states = attn0 + attn1

        hidden_states = residual + hidden_states

        #Fully Connected
       


        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp0 = self.mlp_device0(hidden_states)
        mlp1 = self.mlp_device1(hidden_states)


        hidden_states = mlp0  + mlp1
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights0,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

      
class Multi_Device_LlamaDecoderLayer(nn.Module):
    def __init__(self, oigin_layer:LlamaDecoderLayer  , config: LlamaConfig, layer_idx: int , rank : int ,world_size : int ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx= layer_idx
        self.rank = rank 
        self.world_size = world_size
        self.self_attn= MyCustom_LlamaAttention(oigin_layer.self_attn , rank=self.rank , world_size=self.world_size    )
        self.mlp = MyCustomMLP(oigin_layer.mlp , rank=self.rank , world_size=self.world_size    )
        self.input_layernorm = copy.deepcopy(oigin_layer.input_layernorm )
        self.post_attention_layernorm =copy.deepcopy(oigin_layer.post_attention_layernorm)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # print("decoder fowarding")
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        return outputs
class Single_Device_MyCustom_LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, orig_attn: LlamaAttention, rank: int, world_size: int ):
        super().__init__()
        
        self.config: LlamaConfig = orig_attn.config
        self.attention_dropout = self.config.attention_dropout
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.layer_idx = orig_attn.layer_idx

        self.rank = rank
        self.world_size = world_size 

        self.head_dim =  self.hidden_size // self.num_heads
        # print("self head dim is " , self.head_dim)
        self.num_heads_per_device = self.num_heads // world_size

        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        self.num_key_value_heads =self.config.num_key_value_heads
        self.num_key_value_heads_per_device = self.num_key_value_heads // world_size


        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads_per_device * self.head_dim, bias=self.config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads_per_device * self.head_dim, bias=self.config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads_per_device * self.head_dim, bias=self.config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads_per_device * self.head_dim, self.hidden_size, bias=self.config.attention_bias)
        # rank    = dist.get_rank()
        start_of_query_head = rank * self.num_heads_per_device
        end_of_query_head   = start_of_query_head +  self.num_heads_per_device

        # slice your Q rows
        query_column_start = start_of_query_head * self.head_dim
        query_column_end   = end_of_query_head   * self.head_dim

        query_weight = orig_attn.q_proj.weight.data 
        self.q_proj.weight.data.copy_(query_weight[query_column_start :query_column_end, :]) #column-wise
        # self.q_proj.weight.data.copy_(query_weight[:]) #column-wise

        start_of_key_value_head = rank * self.num_key_value_heads_per_device
        end_of_key_value_head   = start_of_key_value_head +  self.num_key_value_heads_per_device
        key_value_row_start = start_of_key_value_head * self.head_dim
        key_value_row_end   =  end_of_key_value_head   * self.head_dim

        key_weight = orig_attn.k_proj.weight.data 
        self.k_proj.weight.data.copy_(key_weight[ key_value_row_start :key_value_row_end, :]) #column-wise
        # self.k_proj.weight.data.copy_(key_weight[:]) #column-wise
        value_weight = orig_attn.v_proj.weight.data 
        self.v_proj.weight.data.copy_(value_weight[key_value_row_start :key_value_row_end, :]) #column-wise
        o_weight = orig_attn.o_proj.weight.data 
        self.o_proj.weight.data.copy_ (o_weight [ : , query_column_start :query_column_end]) #row-wise
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        if self.config.attention_bias:
            q_b = orig_attn.q_proj.bias.data 
            self.q_proj.bias.data.copy_(q_b[query_column_start:query_column_end])
            k_b = orig_attn.k_proj.bias.data 
            self.k_proj.bias.data.copy_(k_b[query_column_start:query_column_end])
            v_b = orig_attn.v_proj.bias.data 
            self.v_proj.bias.data.copy_(v_b[query_column_start:query_column_end])
            o_b = orig_attn.o_proj.bias.data 
            self.o_proj.bias.data.copy_(o_b)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads_per_device, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_key_value_heads_per_device, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_key_value_heads_per_device, self.head_dim).transpose(1, 2)
        # print("position_ids is " , position_ids)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key = apply_rotary_pos_emb(query_states, key, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            if  position_ids[0][0] != 0 :
                # print(len( past_key_value[self.layer_idx]))
                # print( past_key_value[self.layer_idx][0])
                # print("past_key_value[self.layer_idx][0] is ", past_key_value[self.layer_idx][0])
                # print("past_key_value[self.layer_idx][0] shape is " , past_key_value[self.layer_idx][0].shape )
                past_key = past_key_value[self.layer_idx][0] [: , self.rank * self.num_key_value_heads_per_device :  (self.rank+1) * self.num_key_value_heads_per_device  , :  , : ]
                past_value = past_key_value[self.layer_idx][1 ][ : , self.rank * self.num_key_value_heads_per_device :  (self.rank+1) * self.num_key_value_heads_per_device  ,   :  , : ]
                # print(" before past_key shape is " , past_key.shape )
                # print("key shape is " , key.shape )
                past_key = torch.concatenate( (past_key , key) , dim = 2)
                past_value = torch.concatenate( (past_value , value) , dim = 2)
                # print("after past_key shape is " , past_key.shape )
            else:
                past_key = key
                past_value = value

        key_states = repeat_kv(past_key, self.num_key_value_groups)
        value_states = repeat_kv(past_value, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        
        if causal_mask is not None:
            # print("causal_mask shape ", causal_mask.shape)
            causal_mask = causal_mask[: ,   self.rank * self.num_key_value_heads_per_device :  (self.rank+1) * self.num_key_value_heads_per_device , : , : ] 
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value , key , value ,cos ,sin


class MyCustom_LlamaAttention(nn.Module):
    def __init__(self, orig_attn: LlamaAttention, rank: int, world_size: int ):
        super().__init__()
        
        self.config: LlamaConfig = orig_attn.config
        self.attention_dropout = self.config.attention_dropout
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.layer_idx = orig_attn.layer_idx

        self.rank = rank
        self.world_size = world_size 

        self.head_dim =  self.hidden_size // self.num_heads
        # print("self head dim is " , self.head_dim)
        self.num_heads_per_device = self.num_heads // world_size

        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        self.num_key_value_heads =self.config.num_key_value_heads
        self.num_key_value_heads_per_device = self.num_key_value_heads // world_size


        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads_per_device * self.head_dim, bias=self.config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads_per_device * self.head_dim, bias=self.config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads_per_device * self.head_dim, bias=self.config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads_per_device * self.head_dim, self.hidden_size, bias=self.config.attention_bias)
        # rank    = dist.get_rank()
        start_of_query_head = rank * self.num_heads_per_device
        end_of_query_head   = start_of_query_head +  self.num_heads_per_device

        # slice your Q rows
        query_column_start = start_of_query_head * self.head_dim
        query_column_end   = end_of_query_head   * self.head_dim

        query_weight = orig_attn.q_proj.weight.data 
        self.q_proj.weight.data.copy_(query_weight[query_column_start :query_column_end, :]) #column-wise
        # self.q_proj.weight.data.copy_(query_weight[:]) #column-wise

        start_of_key_value_head = rank * self.num_key_value_heads_per_device
        end_of_key_value_head   = start_of_key_value_head +  self.num_key_value_heads_per_device
        key_value_row_start = start_of_key_value_head * self.head_dim
        key_value_row_end   =  end_of_key_value_head   * self.head_dim

        key_weight = orig_attn.k_proj.weight.data 
        self.k_proj.weight.data.copy_(key_weight[ key_value_row_start :key_value_row_end, :]) #column-wise
        # self.k_proj.weight.data.copy_(key_weight[:]) #column-wise
        value_weight = orig_attn.v_proj.weight.data 
        self.v_proj.weight.data.copy_(value_weight[key_value_row_start :key_value_row_end, :]) #column-wise
        o_weight = orig_attn.o_proj.weight.data 
        self.o_proj.weight.data.copy_ (o_weight [ : , query_column_start :query_column_end]) #row-wise
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        if self.config.attention_bias:
            q_b = orig_attn.q_proj.bias.data 
            self.q_proj.bias.data.copy_(q_b[query_column_start:query_column_end])
            k_b = orig_attn.k_proj.bias.data 
            self.k_proj.bias.data.copy_(k_b[query_column_start:query_column_end])
            v_b = orig_attn.v_proj.bias.data 
            self.v_proj.bias.data.copy_(v_b[query_column_start:query_column_end])
            o_b = orig_attn.o_proj.bias.data 
            self.o_proj.bias.data.copy_(o_b)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads_per_device, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads_per_device, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads_per_device, self.head_dim).transpose(1, 2)
        # print("position_ids is " , position_ids)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()


        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        dist.all_reduce(attn_output, op=dist.ReduceOp.SUM)
        return attn_output, None, past_key_value
  


class Single_Device_MyCustomMLP(nn.Module):
    def __init__(self,         
        orig_mlp: LlamaMLP,
        rank: int,
        world_size: int):
        super().__init__()
        config: LlamaConfig = orig_mlp.config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.intermediate_size_per_device = self.intermediate_size // world_size
        assert  self.intermediate_size_per_device * world_size ==self.intermediate_size , "intermediate_size must be divisible by world_size"

        # create your sharded layers
        self.gate_proj = nn.Linear(self.hidden_size,  self.intermediate_size_per_device, bias=config.mlp_bias)
        self.up_proj   = nn.Linear(self.hidden_size,  self.intermediate_size_per_device, bias=config.mlp_bias)
        self.down_proj = nn.Linear( self.intermediate_size_per_device, self.hidden_size, bias=config.mlp_bias)
        self.act_fn    = ACT2FN[config.hidden_act]

        # now copy & slice the **pretrained** weights & biases from the original LlamaMLP
        # (orig_mlp.gate_proj.weight is [M, D], same for up_proj; down_proj.weight is [D, M])
        column_start = rank * self.intermediate_size_per_device
        column_end   = column_start + self.intermediate_size_per_device

        # Column‐parallel slice on the output dim (dim=0)
        gate_weight = orig_mlp.gate_proj.weight.data    # [M, D]
        up_weight   = orig_mlp.up_proj.weight.data      # [M, D]


        # print(" orig_mlp.gate_proj.weight shpae" ,   orig_mlp.gate_proj.weight.shape)
        # print(" self.gate_proj.weight shpae" ,   self.gate_proj.weight.shape)
        self.gate_proj.weight.data.copy_(gate_weight[column_start:column_end, :])
        self.up_proj.weight.data.copy_(up_weight[column_start:column_end, :])

        if config.mlp_bias:
            gate_b = orig_mlp.gate_proj.bias.data   # [M]
            up_b   = orig_mlp.up_proj.bias.data     # [M]
            self.gate_proj.bias.data.copy_(gate_b[column_start:column_end])
            self.up_proj.bias.data.copy_(up_b[column_start:column_end])

        # Row‐parallel slice on the input dim (dim=1)
        row_start = column_start
        row_end   = column_end

        # print("orig_mlp.down_proj shpae" ,  orig_mlp.down_proj.weight.shape)
        # print("self.down_proj shpae" ,  self.down_proj.weight.shape)
        # print("d start is " , d_start)
        # print("d_end is " , d_end)
        down_weight = orig_mlp.down_proj.weight.data    # [D, M]
        self.down_proj.weight.data.copy_(down_weight[:, row_start:row_end])

        if config.mlp_bias:
            # full bias is [D], broadcast to every rank
            down_b = orig_mlp.down_proj.bias.data 
            self.down_proj.bias.data.copy_(down_b)


    def forward(self, x):


        # print("start foward MLP")
        gate_act = self.act_fn(self.gate_proj(x))        # → [B, S, M_per]
        up_out   = self.up_proj(x)                       # → [B, S, M_per]
        hidden   = gate_act * up_out                     # → [B, S, M_per]

        # print("hidden shape" , hidden.shape)

        partial  = self.down_proj(hidden)                # → [B, S, D]

        # print("partial shape" , partial.shape)

        # partial = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # sum across ranks to recompose the full MLP output
        # dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial
    


class MyCustomMLP(nn.Module):
    def __init__(self,         
        orig_mlp: LlamaMLP,
        rank: int,
        world_size: int):
        super().__init__()
        config: LlamaConfig = orig_mlp.config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.intermediate_size_per_device = self.intermediate_size // world_size
        assert  self.intermediate_size_per_device * world_size ==self.intermediate_size , "intermediate_size must be divisible by world_size"

        # create your sharded layers
        self.gate_proj = nn.Linear(self.hidden_size,  self.intermediate_size_per_device, bias=config.mlp_bias)
        self.up_proj   = nn.Linear(self.hidden_size,  self.intermediate_size_per_device, bias=config.mlp_bias)
        self.down_proj = nn.Linear( self.intermediate_size_per_device, self.hidden_size, bias=config.mlp_bias)
        self.act_fn    = ACT2FN[config.hidden_act]

        # now copy & slice the **pretrained** weights & biases from the original LlamaMLP
        # (orig_mlp.gate_proj.weight is [M, D], same for up_proj; down_proj.weight is [D, M])
        column_start = rank * self.intermediate_size_per_device
        column_end   = column_start + self.intermediate_size_per_device

        # Column‐parallel slice on the output dim (dim=0)
        gate_weight = orig_mlp.gate_proj.weight.data    # [M, D]
        up_weight   = orig_mlp.up_proj.weight.data      # [M, D]


        # print(" orig_mlp.gate_proj.weight shpae" ,   orig_mlp.gate_proj.weight.shape)
        # print(" self.gate_proj.weight shpae" ,   self.gate_proj.weight.shape)
        self.gate_proj.weight.data.copy_(gate_weight[column_start:column_end, :])
        self.up_proj.weight.data.copy_(up_weight[column_start:column_end, :])

        if config.mlp_bias:
            gate_b = orig_mlp.gate_proj.bias.data   # [M]
            up_b   = orig_mlp.up_proj.bias.data     # [M]
            self.gate_proj.bias.data.copy_(gate_b[column_start:column_end])
            self.up_proj.bias.data.copy_(up_b[column_start:column_end])

        # Row‐parallel slice on the input dim (dim=1)
        row_start = column_start
        row_end   = column_end

        # print("orig_mlp.down_proj shpae" ,  orig_mlp.down_proj.weight.shape)
        # print("self.down_proj shpae" ,  self.down_proj.weight.shape)
        # print("d start is " , d_start)
        # print("d_end is " , d_end)
        down_weight = orig_mlp.down_proj.weight.data    # [D, M]
        self.down_proj.weight.data.copy_(down_weight[:, row_start:row_end])

        if config.mlp_bias:
            # full bias is [D], broadcast to every rank
            down_b = orig_mlp.down_proj.bias.data 
            self.down_proj.bias.data.copy_(down_b)


    def forward(self, x):


        # print("start foward MLP")
        gate_act = self.act_fn(self.gate_proj(x))        # → [B, S, M_per]
        up_out   = self.up_proj(x)                       # → [B, S, M_per]
        hidden   = gate_act * up_out                     # → [B, S, M_per]

        # print("hidden shape" , hidden.shape)

        partial  = self.down_proj(hidden)                # → [B, S, D]

        # print("partial shape" , partial.shape)

        # partial = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # sum across ranks to recompose the full MLP output
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial

class My_LMHEAD(nn.Module):
    def __init__(self, orig_lm_head: nn.Linear, rank: int, world_size: int):
        super().__init__()
        self.in_features = orig_lm_head.in_features
        self.out_features = orig_lm_head.out_features
        self.rank = rank
        self.world_size = world_size

        # Shard vocab along output (row) dimension
        self.out_features_per_device =  self.out_features // self.world_size

        assert  self.out_features_per_device * self.world_size ==self.out_features , "out_features must be divisible by world_size"
        column_start = self.rank * self.out_features_per_device
        column_end   = column_start + self.out_features_per_device



        

        # Create local linear layer
        self.lm_head = nn.Linear(self.in_features, self.out_features_per_device , bias=orig_lm_head.bias is not None)

        # Copy weights
        
        self.lm_head.weight.data.copy_(orig_lm_head.weight[column_start:column_end, :])
        if orig_lm_head.bias is not None:
            self.lm_head.bias.data.copy_(orig_lm_head.bias[column_start:column_end])

    def forward(self, hidden_states):
        # hidden_states: [B, T, H]
        last_hidden = hidden_states[:, -1, :] 
        local_logits = self.lm_head(last_hidden)
        logits_list = [torch.empty_like(local_logits) for _ in range(self.world_size)]

        # Run all_gather
        dist.all_gather(logits_list, local_logits)
        
        return torch.cat(logits_list, dim=-1)


def multi_replace_LlamaDecoderLayer(model: torch.nn.Module , rank:int , world_size: int):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LlamaDecoderLayer):
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            tp_decoder = Multi_Device_LlamaDecoderLayer(mod, model.config  , mod.self_attn.layer_idx , rank=rank , world_size= world_size)
            setattr(parent, attr, tp_decoder)
def single_replace_LlamaDecoderLayer(model: torch.nn.Module):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LlamaDecoderLayer):
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            tp_decoder = Single_Device_LlamaDecoderLayer(mod, model.config  , mod.self_attn.layer_idx)

            setattr(parent, attr, tp_decoder)
def replace_llama_attn(model: torch.nn.Module, rank: int, world_size: int):
    for name, mod in list(model.named_modules()):
        print("name is " , name)
        if isinstance(mod, LlamaAttention):
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)

            
            # tp_attn = MyCustom_LlamaAttention(mod, rank, world_size)

            tp_attn = MyCustom_LlamaAttention(mod, rank, world_size)
            # copy weights
            # tp_mlp.gate_proj.load_state_dict(mod.gate_proj.state_dict())
            # tp_mlp.up_proj  .load_state_dict(mod.up_proj  .state_dict())
            # tp_mlp.down_proj.load_state_dict(mod.down_proj.state_dict())
            setattr(parent, attr, tp_attn)

def replace_llama_mlps(model: torch.nn.Module, rank: int, world_size: int):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LlamaMLP):
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            tp_mlp = MyCustomMLP(mod, rank, world_size)
            # copy weights
            # tp_mlp.gate_proj.load_state_dict(mod.gate_proj.state_dict())
            # tp_mlp.up_proj  .load_state_dict(mod.up_proj  .state_dict())
            # tp_mlp.down_proj.load_state_dict(mod.down_proj.state_dict())
            setattr(parent, attr, tp_mlp)
import time

def main():
    # torchrun will set these ENV vars for us:
    USE_TEMPERATURE_SAMPLING = True
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 1) init process group for inference
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    # 2) load & patch model


    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")




    # replace_llama_mlps(model, rank, world_size)
    # replace_llama_attn(model, rank, world_size)
    multi_replace_LlamaDecoderLayer(model , rank ,  world_size )

    model.lm_head = My_LMHEAD(model.lm_head, rank, world_size).to(model.device)
    device = torch.device(f"cuda:{local_rank}")
    print("rank is ", rank)
    print("world_size is ", world_size)
    # 3) move to GPU, eval mode
    model.cuda(local_rank)
    model.eval()

    # 4) prepare your prompt
    
    #inputs = tokenizer("Please summarize this introduction and as long as possible: In October 1886, Scotsman David Danskin and fifteen fellow munitions workers in Woolwich formed the Dial Square Football Club, named after a workshop at the heart of the Royal Arsenal complex. Each member contributed sixpence, and Danskin also added three shillings to help form the club. Dial Square played their first match on 11 December 1886 against the Eastern Wanderers and won 6–0. The club had been renamed Royal Arsenal by January 1887,[15][16] and its first home was Plumstead Common,[15] though they spent most of their time playing at the Manor Ground. Their first trophies were the Kent Senior Cup and London Charity Cup in 1889–90 and the London Senior Cup in 1890–91; these were the only county association trophies Arsenal won during their time in South East London.[17][18] In 1891, Royal Arsenal became the first London club to turn professional.[19] Royal Arsenal was renamed for the second time upon becoming a limited liability company in 1893. They registered their new name, Woolwich Arsenal, with the Football League when the club ascended later that year.[20][21]: 5–21  Woolwich Arsenal was the first southern member of the Football League, starting out in the Second Division and reaching the First Division in 1904. Falling attendances, due to financial difficulties among the munitions workers and the arrival of more accessible football clubs elsewhere in the city, led the club close to bankruptcy by 1910.[22][21]: 112–149  Businessmen Henry Norris and William Hall became involved in the club, and sought to move them elsewhere.[23][21]: 22–42 1912–1925: Bank of England club In 1913, soon after relegation back to the Second Division, the club moved across the river to the new Arsenal Stadium in Highbury.[24][25][26] In 1919, the Football League controversially voted to promote The Arsenal, instead of relegated local rivals Tottenham Hotspur, into the newly enlarged First Division, despite only finishing fifth in the Second Division's last pre-war season of 1914–15. Later that year, The Arsenal started dropping The in official documents, gradually shifting its name for the final time towards Arsenal, as it is generally known today. With a new home and First Division football, attendances were more than double those at the Manor Ground, and Arsenal's budget grew rapidly.[28][29] With record-breaking spending and gate receipts, Arsenal quickly became known as the Bank of England club. Arsenal's location and record-breaking salary offer lured star Huddersfield Town manager Herbert Chapman in 1925.[32][33] Over the next five years, Chapman built a revolutionary new Arsenal. Firstly, he appointed an enduring new trainer, Tom Whittaker who would one day rise to become a fabled Arsenal manager himself.[34] With the help of player Charlie Buchan, implemented the nascent WM formation which would serve as a stable bedrock to his outfit.[35][36] He also captured generational young talents such as Cliff Bastin and Eddie Hapgood, whilst also lavishing Highbury's high income on stars such as David Jack and Alex James. Transformed, Chapman's Arsenal claimed their first national trophy, the FA Cup in 1930, and League Championships followed in 1930–31 and 1932–33.[37] Chapman also presided over off-pitch changes: white sleeves and shirt numbers were added to the kit;[note 2] a Tube station was named after the club;[41][42] and the first of two opulent Art Deco stands was completed, with some of the first floodlights in English football.[29] Suddenly, in the middle of the 1933–34 season, Chapman died of pneumonia. 1934–1947: Shaw, Allison and the Second World War Chapman's death meant work was left to his colleagues Joe Shaw and George Allison, with both proving to be shrewd and consummate custodians of Chapman's excellent Arsenal team, seeing out a hat-trick of league wins with the 1933–34, 1934–35, and 1937–38 titles, and then furthermore winning the 1936 FA Cup.[44][45] World War II meant the Football League was suspended for seven years. While Arsenal were paraded by the nation as a symbol of solidarity with war efforts, the war took a huge toll on the team as the club had had more players killed than any top flight club.[46] Furthermore, debt from reconstructing an ambitious North Bank Stand redevelopment greatly bled Arsenal's resources.1947–1962: Tom Whittaker's meteoric Gunners Despite this period of turbulence and churn, Arsenal returned to win the league in the second post-war season of 1947–48. This was Tom Whittaker's first season as manager, and meant the club equalled the champions of England record.[3] Whittaker, despite his disarming humble and modest disposition, was oft-referred to as the brains behind charismatic Chapman's legendary Arsenal side.[48][49] He gathered a successful and highly skilled Arsenal side in spite of greatly limited resources, with a fiery and expansive style that drove great fanfare at the time.[50] They won a third FA Cup in 1950, and then won a record-breaking seventh championship in 1952–53 making Arsenal the most successful team in English history at the time. Arsenal were not to win the League or the FA Cup for another 18 years. The '53 Champions squad had aged, and the club failed to attract strong enough replacements.[53] Although Arsenal were competitive during these years, their fortunes had waned; the club spent most of the 1950s and 1960s in mid-table mediocrity.[54] Even former England captain Billy Wright could not bring the club any success as manager, in a stint between 1962 and 1966. Arsenal tentatively appointed club physiotherapist Bertie Mee as acting manager in 1966.[56][57] With new assistant Don Howe and new players such as Bob McNab and George Graham, Mee led Arsenal to their first League Cup finals, in 1967–68 and 1968–69. Next season saw a breakthrough, with Arsenal's first competitive European trophy, the 1969–70 Inter-Cities Fairs Cup. The season after, Arsenal achieved an even greater triumph with their first League and FA Cup double, and a new champions of England record.[58] This marked a premature high point of the decade; the Double-winning side was soon broken up and the rest of the decade was characterised by a series of near misses, with Arsenal finishing as FA Cup runners up in 1972, and First Division runners-up in 1972–73.Former player Terry Neill succeeded Mee in 1976. At the age of 34, he became the youngest Arsenal manager to date.[59] With new signings like Malcolm Macdonald and Pat Jennings, and a crop of talent in the side like Liam Brady and Frank Stapleton, the club reached a trio of FA Cup finals (1978 FA Cup, 1979 FA Cup and 1980 FA Cup), and lost the 1980 European Cup Winners' Cup Final on penalties. The club's only trophy during this time was the 1979 FA Cup, achieved with a last-minute 3–2 victory over Manchester United, in a final is widely regarded as a classic.One of Mee's double winners, George Graham, returned as manager in 1986, with Arsenal winning their first League Cup in 1987, Graham's first season in charge. New signings Nigel Winterburn, Lee Dixon and Steve Bould had joined the club by 1988 to complete the famous Back Four, led by homegrown player Tony Adams.[62][note 3] Graham's credo of prioritising defensive excellence seemingly clashed with the club's traditionally expansive motifs and many had skepticism whether it would work with the young squad at the club in that time period; however, his methods quickly gained a cult following after initial successes. The side immediately won the 1988 Football League Centenary Trophy, and followed it with the 1988–89 Football League title, snatched with a last-minute goal in the final game of the season against fellow title challengers Liverpool.[64] Graham's Arsenal won another title in 1990–91, losing only one match, won the FA Cup and League Cup double in 1993, and the European Cup Winners' Cup in 1994. Graham's reputation was tarnished when he was found to have taken kickbacks from agent Rune Hauge for signing certain players, and he was dismissed in 1995.[65][66] His replacement, Bruce Rioch, lasted for only one season, leaving the club after a dispute with the board of directors. The club metamorphosed during the tenure of French manager Arsène Wenger, who was appointed in 1996. Attacking football,[68] an overhaul of dietary and fitness practices,[note 4] and elite scouting[note 5] defined his reign. Accumulating key players from Wenger's homeland, such as Patrick Vieira and Thierry Henry, Arsenal won a second League and Cup double in 1997–98 and a third in 2001–02. In addition, the club reached the final of the 1999–2000 UEFA Cup, were victorious in the 2003 and 2005 FA Cup finals, and won the Premier League in 2003–04 without losing a single match, an achievement which earned the side the nickname The Invincibles.[77] This feat came within a run of 49 league matches unbeaten from 7 May 2003 to 24 October 2004, a national record. Arsenal finished in either first or second place in the league in eight of Wenger's first nine seasons at the club, although they never won the title in two consecutive seasons.[79] The club had never progressed beyond the quarter-finals of the Champions League until 2005–06; in that season, they became the first club from London to reach the final in the competition's fifty-year history, but were beaten 2–1 by Barcelona.[80] In July 2006, they moved into the Emirates Stadium, after 93 years at Highbury.[81] Arsenal reached the finals of the 2007 and 2011 League Cups, losing 2–1 to Chelsea and Birmingham City respectively. The club had not gained a trophy since the 2005 FA Cup until, spearheaded by club record acquisition Mesut Özil, Arsenal beat Hull City in the 2014 FA Cup Final, coming back from a 2–0 deficit to win the match 3–2.[82] A year later, Arsenal completed another victorious FA Cup campaign,[83] and became the most successful club in the tournament's history by winning their 13th FA Cup in 2016–17. However, in that same season Arsenal finished fifth in the league, the first time they had finished outside the top four since before Wenger arrived in 1996.[84] In his 21st and final season, Arsenal under Arsene Wenger finished sixth and won the FA Community Shield.[85][86] Wenger departed Arsenal following the end of the season on 13 May 2018. After conducting an overhaul in the club's operating model to coincide with Wenger's departure, Spaniard Unai Emery was named as the club's new head coach on 23 May 2018. He became the club's first ever 'head coach' and second manager from outside the United Kingdom.[88][89] In Emery's first season, Arsenal finished fifth in the Premier League and as runner-up in the Europa League.[90][91] On 29 November 2019, Emery was dismissed as manager and former player and assistant first team coach Freddie Ljungberg was appointed as interim head coach. ", return_tensors="pt").to(local_rank)


    inputs = tokenizer("Please give me a story?", return_tensors="pt").to(local_rank)
    if(USE_TEMPERATURE_SAMPLING == True):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # 5) inference: inside MyCustomMLP.forward you already all_reduce partials
        # with torch.no_grad():
        #     outputs = model.generate(**inputs, max_new_tokens=2000 , temperature=0.0, do_sample=False ,  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id  )
        temperature = 0.7
        max_new_tokens=4000
        past_kv = None
        generated = inputs["input_ids"]
        with torch.no_grad():
            outputs = model(
                    input_ids=generated,  # only feed last token
                    past_key_values=past_kv,
                    use_cache=True
                )
            logits = outputs.logits
            past_kv = outputs.past_key_values
            if rank == 0:
                logits = logits / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) 
            else:
                next_token = torch.empty((logits.shape[0], 1), dtype=torch.long, device=device)
            dist.broadcast(next_token, src=0)
            generated = torch.cat([generated, next_token], dim=-1)

        
        for step in range(max_new_tokens-1):  
            with torch.no_grad():
                outputs = model(
                    input_ids=generated[:, -1:], 
                    past_key_values=past_kv,
                    use_cache=True
                )
            logits = outputs.logits
            past_kv = outputs.past_key_values
            if rank == 0:
                logits = logits / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) 
            else:
                next_token = torch.empty((logits.shape[0], 1), dtype=torch.long, device=device)
            dist.broadcast(next_token, src=0)
            generated = torch.cat([generated, next_token], dim=-1)
            if (next_token == tokenizer.eos_token_id).all():
                break
        torch.cuda.synchronize()

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

        if rank ==0 :
            print(tokenizer.decode(generated[0], skip_special_tokens=True))
            print(f"Inference time: {elapsed_time_ms:.3f} ms")

        dist.destroy_process_group()
    else:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # 5) inference: inside MyCustomMLP.forward you already all_reduce partials
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=2000 , temperature=0.0, do_sample=False ,  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id  )
        torch.cuda.synchronize()
        end_event.record()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        
        # 6) only rank 0 needs to decode / print
        if rank ==0 :
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            print(f"Inference time: {elapsed_time_ms:.3f} ms")

        # 7) clean up
        dist.destroy_process_group()

if __name__ == "__main__":
    main()