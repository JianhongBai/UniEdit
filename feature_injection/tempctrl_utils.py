import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import save_image
from einops import rearrange, repeat
from lavie_models.attention import RelativePositionBias

class TempAttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.gamma = 1.0
        self.gamma_new = 1.0
        self.time_rel_pos_bias_num = 16

    def after_step(self):
        pass

    def __call__(self, query, key, value, attention_mask, time_rel_pos_bias):
        out = self.forward(query, key, value, attention_mask, time_rel_pos_bias)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, query, key, value, attention_mask, time_rel_pos_bias):
        attention_scores = torch.einsum('... h i d, ... h j d -> ... h i j', query, key)
        attention_scores = attention_scores + time_rel_pos_bias

        if attention_mask is not None:
            # add attention mask
            attention_scores = attention_scores + attention_mask

        attention_scores = attention_scores - attention_scores.amax(dim = -1, keepdim = True).detach()

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # print(attention_probs[0][0])

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output 
        hidden_states = torch.einsum('... h i j, ... h j d -> ... h i d', attention_probs, value)
        hidden_states = rearrange(hidden_states, 'b h f d -> b f (h d)')
        hidden_states *= self.gamma
        return hidden_states

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

def exists(x):
    return x is not None

def regiter_temp_attention_editor_diffusers(model, editor: TempAttentionBase):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            time_rel_pos_bias = self.time_rel_pos_bias(hidden_states.shape[1], device=hidden_states.device)
            batch_size, sequence_length, _ = hidden_states.shape

            encoder_hidden_states = encoder_hidden_states

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states) # [b (h w)] f (nd * d)
            dim = query.shape[-1]
            
            if self.added_kv_proj_dim is not None:
                key = self.to_k(hidden_states)
                value = self.to_v(hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)
                encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
                encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

                key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
                value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)
                
            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            # attention, what we cannot get enough of
            assert not self._use_memory_efficient_attention_xformers
            assert self._slice_size is None or query.shape[0] // self._slice_size == 1

            if self.upcast_attention:
                query = query.float()
                key = key.float()

            # reshape for adding time positional bais
            query = self.scale * rearrange(query, 'b f (h d) -> b h f d', h=self.heads) # d: dim_head; n: heads
            key = rearrange(key, 'b f (h d) -> b h f d', h=self.heads) # d: dim_head; n: heads
            value = rearrange(value, 'b f (h d) -> b h f d', h=self.heads) # d: dim_head; n: heads

            if exists(self.rotary_emb):
                query = self.rotary_emb.rotate_queries_or_keys(query)
                key = self.rotary_emb.rotate_queries_or_keys(key)

            hidden_states = editor(query, key, value, attention_mask, time_rel_pos_bias)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states * 1.

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'TemporalAttention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count