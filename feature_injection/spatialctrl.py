import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .spatialctrl_utils import AttentionBase

class SpatialAttentionWithMask(AttentionBase):
    def __init__(self, mask_s=None, mask_t=None, ratio=1.6, mask_save_dir=None):
        super().__init__()
        self.layer_idx = list(range(8, 16))
        self.step_idx = list(range(4, 50))
        self.perform_masked_attn = False

        self.struc_ctrl_step_idx = []
        self.struc_ctrl_layer_idx = []
        self.content_pre_step_idx = []
        self.content_pre_layer_idx = []
        self.bg_enhance_step_idx = []
        self.bg_enhance_layer_idx = []
        self.mask_s = mask_s
        self.mask_t = mask_t
        self.hw_ratio = ratio

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1] / 1.6))  # 512/320=1.6
        W = int(1.6 * W)
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            print("masked attention")
            mask = self.mask_s.unsqueeze(0).unsqueeze(0)
            mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
            mask = mask.flatten()
            # background
            sim_bg = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            # object
            sim_fg = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim = torch.cat([sim_fg, sim_bg], dim=0)
        attn = sim.softmax(-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
        return out
    
    def attn_modify(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        H = W = int(np.sqrt(q.shape[1] / 1.6))  # 512/320=1.6
        W = int(1.6 * W)
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * kwargs.get("scale")
        if kwargs.get("is_mask_attn") and self.mask_s is not None:
            new_sim_bg = torch.zeros_like(sim)
            new_sim_fg = torch.zeros_like(sim)
            head_num = 8
            for i, mask_s in enumerate(self.mask_s):
                mask = mask_s.unsqueeze(0).unsqueeze(0)
                mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
                mask = mask.flatten()

                # background
                new_sim_bg[head_num*i : head_num*(i+1)] = sim[head_num*i : head_num*(i+1)] + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
                # object
                new_sim_fg[head_num*i : head_num*(i+1)] = sim[head_num*i : head_num*(i+1)] + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min)
            sim = torch.cat([new_sim_fg, new_sim_bg], dim=0)
        
        attn = sim.softmax(dim=-1)
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if self.motion_editing:
            if is_cross:
                return AttentionBase().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

            H = W = int(np.sqrt(q.shape[1] / self.hw_ratio))  # 512/320=1.6
            W = int(self.hw_ratio * W)
            ku, kc = k.chunk(2)
            qu, qc = q.chunk(2)
            vu, vc = v.chunk(2)
            N = ku.shape[0] // 3

            sim = None
            attn = None
            attnu = None
            attnc = None
            if not is_cross and self.cur_step in self.struc_ctrl_step_idx and self.cur_att_layer // 2 in self.struc_ctrl_layer_idx:
                ku = ku[:N].repeat(3, 1, 1)
                kc = kc[:N].repeat(3, 1, 1)
                qu = qu[:N].repeat(3, 1, 1)
                qc = qc[:N].repeat(3, 1, 1)

            if not is_cross and self.cur_step in self.content_pre_step_idx and self.cur_att_layer // 2 in self.content_pre_layer_idx:
                mask = self.mask_s_fg.unsqueeze(1)
                mask = F.interpolate(mask, (H, W)).unsqueeze(1)
                mask = mask.view(16, 1, -1).unsqueeze(-1)
                mask = mask.repeat(1, 8, 1, qu[2*N:].shape[2])
                mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
                mask = mask.expand_as(qu[2*N:3*N]).bool()
                vu[N:2*N][~mask] = vu[:N][~mask]
                vc[N:2*N][~mask] = vc[:N][~mask]
                ku[N:2*N][~mask] = ku[:N][~mask]
                kc[N:2*N][~mask] = kc[:N][~mask]

                mask = self.mask_t.unsqueeze(1)
                mask = F.interpolate(mask, (H, W)).unsqueeze(1)
                mask = mask.view(16, 1, -1).unsqueeze(-1)
                mask = mask.repeat(1, 8, 1, qu[2*N:].shape[2])
                mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
                mask = mask.expand_as(qu[2*N:3*N]).bool()
                qu[N:2*N][mask] = qu[2*N:][mask]
                qc[N:2*N][mask] = qc[2*N:][mask]

            out_u_123 = self.attn_modify(qu, ku, vu, sim, attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_123 = self.attn_modify(qc, kc, vc, sim, attnc, is_cross, place_in_unet, num_heads, **kwargs)

            out_u_target = self.attn_modify(qu[N:2*N], ku[N:2*N], vu[N:2*N], None, attnu, is_cross, place_in_unet, num_heads, is_mask_attn=self.perform_masked_attn, **kwargs)
            out_c_target = self.attn_modify(qc[N:2*N], kc[N:2*N], vc[N:2*N], None, attnc, is_cross, place_in_unet, num_heads, is_mask_attn=self.perform_masked_attn, **kwargs)

            if self.mask_s is not None and self.mask_t is not None and self.perform_masked_attn:
                out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
                out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)

                mask = F.interpolate(self.mask_t.unsqueeze(1), (H, W))
                mask = mask.squeeze(1)  # (F, h, w)
                mask = mask.reshape(mask.shape[0], -1, 1)  # (F, hw, 1)

                out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
                out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

            out_u_123[out_u_123.shape[0] // 3:2*out_u_123.shape[0] // 3] = out_u_target
            out_c_123[out_c_123.shape[0] // 3:2*out_c_123.shape[0] // 3] = out_c_target
            out = torch.cat([out_u_123, out_c_123], dim=0)
        else:
            if is_cross:
                return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

            H = W = int(np.sqrt(q.shape[1] / self.hw_ratio))  # 512/320
            W = int(self.hw_ratio * W)
            qu, qc = q.chunk(2)
            ku, kc = k.chunk(2)
            vu, vc = v.chunk(2)
            sim = None
            attn = None
            attnu = None
            attnc = None
            N = ku.shape[0] // 2
            if not is_cross and self.cur_step in self.struc_ctrl_step_idx and self.cur_att_layer // 2 in self.struc_ctrl_layer_idx:
                qu = qu[:N].repeat(2, 1, 1)
                qc = qc[:N].repeat(2, 1, 1)
                ku = ku[:N].repeat(2, 1, 1)
                kc = kc[:N].repeat(2, 1, 1)

            if not is_cross and self.cur_step in self.content_pre_step_idx and self.cur_att_layer // 2 in self.content_pre_layer_idx:
                vu[N:2*N] = vu[:N]
                vc[N:2*N] = vc[:N]
            
            if not is_cross and self.cur_step in self.bg_enhance_step_idx and self.cur_att_layer // 2 in self.bg_enhance_layer_idx:
                mask = self.mask_s.unsqueeze(1)
                mask = F.interpolate(mask, (H, W)).unsqueeze(1)
                mask = mask.view(16, 1, -1).unsqueeze(-1) # (F, 1, H*W, 1)
                mask = mask.repeat(1, 8, 1, qu[N:].shape[2])
                mask = mask.view(-1, mask.shape[-2], mask.shape[-1])
                mask = mask.expand_as(qu[N:2*N]).bool()
                vu[N:2*N][~mask] = vu[:N][~mask]
                vc[N:2*N][~mask] = vc[:N][~mask]

            out_u_source = self.attn_modify(qu[:N], ku[:N], vu[:N], None, attnu, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_source = self.attn_modify(qc[:N], kc[:N], vc[:N], None, attnc, is_cross, place_in_unet, num_heads, **kwargs)

            out_u_target = self.attn_modify(qu[N:], ku[N:], vu[N:], None, attnu, is_cross, place_in_unet, num_heads, is_mask_attn=self.perform_masked_attn, **kwargs)
            out_c_target = self.attn_modify(qc[N:], kc[N:], vc[N:], None, attnc, is_cross, place_in_unet, num_heads, is_mask_attn=self.perform_masked_attn, **kwargs)

            if self.perform_masked_attn:
                out_u_target_fg, out_u_target_bg = out_u_target.chunk(2, 0)
                out_c_target_fg, out_c_target_bg = out_c_target.chunk(2, 0)
                mask = F.interpolate(self.mask_t.unsqueeze(0).unsqueeze(0), (H, W))
                mask = mask.reshape(-1, 1) 
                out_u_target = out_u_target_fg * mask + out_u_target_bg * (1 - mask)
                out_c_target = out_c_target_fg * mask + out_c_target_bg * (1 - mask)

            out = torch.cat([out_u_source, out_u_target, out_c_source, out_c_target], dim=0)
        return out
