import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .tempctrl_utils import TempAttentionBase
from torchvision.utils import save_image
import torch.nn as nn

class TempAttentionWithMask(TempAttentionBase):
    def __init__(self, mask_t=None, ratio=1.6):
        super().__init__()
        self.mask_t = mask_t
        self.perform_masked_attn = False
        self.hw_ratio = ratio
    
    def attn_modify(self, q, k, v, attention_mask, time_rel_pos_bias, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        if kwargs.get("is_mask_attn"):
            bHhw = q.shape[0] * 6
        else:
            bHhw = q.shape[0]
        H = W = int(np.sqrt(bHhw / 6 / self.hw_ratio))
        W = int(self.hw_ratio * W) 
        sim = torch.einsum('... h i d, ... h j d -> ... h i j', q, k) # * kwargs.get("scale")
        sim = sim + time_rel_pos_bias
        if attention_mask is not None:
            # add attention mask
            sim = sim + attention_mask
        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            # print("masked attention")
            mask = self.mask_s.unsqueeze(1)
            mask = F.interpolate(mask, (H, W))
            mask = mask.permute(2, 3, 1, 0)
            mask = mask.reshape(-1, 1, 1, 16)

            # background
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            # object
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg], dim=0)
        
        attn = sim.softmax(dim=-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)

        out = torch.einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, 'b h f d -> b f (h d)')
        return out

    def forward(self, q, k, v, attention_mask, time_rel_pos_bias, **kwargs):
        """
        Attention forward function
        """

        H = W = int(np.sqrt(q.shape[0] / 6 / self.hw_ratio)) 
        W = int(self.hw_ratio * W)
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        N = ku.shape[0] // 3
        if self.mask_t is not None:
            mask = self.mask_t.unsqueeze(1)
            mask = F.interpolate(mask, (H, W))
            mask = mask.permute(2, 3, 1, 0)
            mask = mask.view(-1, 1, 16).unsqueeze(-1).bool()   # (H*W, 1, 16, 1)
            mask = mask.expand_as(qu[N:2*N])
            qu[N:2*N][mask] = qu[2*N:][mask]
            qc[N:2*N][mask] = qc[2*N:][mask]
            ku[N:2*N][mask] = ku[2*N:][mask]
            kc[N:2*N][mask] = kc[2*N:][mask]
        else:
            qu = torch.cat([qu[:N], qu[2*N:].repeat(2, 1, 1, 1)], dim=0)
            qc = torch.cat([qc[:N], qc[2*N:].repeat(2, 1, 1, 1)], dim=0)
            ku = torch.cat([ku[:N], ku[2*N:].repeat(2, 1, 1, 1)], dim=0)
            kc = torch.cat([kc[:N], kc[2*N:].repeat(2, 1, 1, 1)], dim=0)

        query = torch.cat([qu, qc], dim=0)
        key = torch.cat([ku, kc], dim=0)
        value = torch.cat([vu, vc], dim=0)

        out_uc_123 = self.attn_modify(query, key, value, attention_mask, time_rel_pos_bias, **kwargs)

        if self.mask_t is not None and self.perform_masked_attn:

            out_u_2 = self.attn_modify(qu[N:2*N], ku[N:2*N], vu[N:2*N], attention_mask, time_rel_pos_bias, is_mask_attn=self.perform_masked_attn, **kwargs)
            out_c_2 = self.attn_modify(qc[N:2*N], kc[N:2*N], vc[N:2*N], attention_mask, time_rel_pos_bias, is_mask_attn=self.perform_masked_attn, **kwargs)
            out_u_target_fg, out_u_target_bg = out_u_2.chunk(2, 0)
            out_c_target_fg, out_c_target_bg = out_c_2.chunk(2, 0)

            mask = F.interpolate(self.mask_t.unsqueeze(1), (H, W))
            mask = mask.squeeze(1).reshape(mask.shape[0], -1).permute(1, 0).unsqueeze(2)
            
            out_u_2 = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
            out_c_2 = out_c_target_fg * mask + out_c_target_bg * (1 - mask)
            out_uc_123[N:2*N] = out_u_2
            out_uc_123[4*N:5*N] = out_c_2
        return out_uc_123