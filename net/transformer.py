from functools import partial
import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
            att_mask = None
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.att_mask = att_mask
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def rope_pos(self, x, pos):
        B, nhead, N, dimx = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, N, dimx*nhead)
        dim = pos.shape[-1]
        sinx, cosx,  siny, cosy= pos.split((dim//4, dim//4, dim//4, dim//4), -1)
        x1, x2, y1, y2 = x[..., 0::4], x[..., 1::4], x[..., 2::4], x[..., 3::4]

        xpos = torch.concatenate([x1 * cosx - x2 * sinx, x2 * cosx + x1 * sinx], dim=-1)
        ypos = torch.concatenate([y1 * cosy - y2 * siny, y2 * cosy + y1 * siny], dim=-1)

        return torch.concatenate([xpos, ypos], dim=-1).reshape(B, N, dimx, nhead).permute(0, 3, 1, 2)
    def forward(self, x, pclsp_embed):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = self.rope_pos(q, pclsp_embed)
        k = self.rope_pos(k, pclsp_embed)
        # B, nhead, N, dimx = v.shape
        # v = v + pclsp_embed.reshape(B, N, dimx, nhead).permute(0, 3, 1, 2)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if self.att_mask is not None:
                attn = attn.masked_fill(self.att_mask[:, :, :, :] == 0, -1e4)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma



class Transformer(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            att_mask = None,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.att_mask = att_mask
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            att_mask=self.att_mask
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pclsp_embed):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), pclsp_embed)))

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

def get_subsequent_mask(seq, sizeM, PointNms):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()

    att_mask = torch.ones((1, len_s, len_s), device=seq.device).bool()
    _, h, w = att_mask.shape
    att_mask = att_mask.view(1, 1, h, w)
    # att_mask[0, 0, :sizeM, sizeM:] = False
    # att_mask[0, 0, sizeM:sizeM + 1 * PointNms, sizeM + 1 * PointNms:] = False
    #
    # att_mask[0, 0, sizeM + 1 * PointNms:sizeM + 2 * PointNms, sizeM:sizeM + 1 * PointNms] = False
    # att_mask[0, 0, sizeM + 1 * PointNms:sizeM + 2 * PointNms, sizeM + 2 * PointNms:] = False
    #
    # att_mask[0, 0, sizeM + 2 * PointNms:sizeM + 3 * PointNms, sizeM:sizeM + 2 * PointNms] = False
    # att_mask[0, 0, sizeM + 2 * PointNms:sizeM + 3 * PointNms, sizeM + 3 * PointNms:] = False
    #
    # att_mask[0, 0, sizeM + 3 * PointNms:sizeM + 4 * PointNms, sizeM:sizeM + 3 * PointNms] = False

    return att_mask
        # self.inputT = torch.ones((1, cfg.TeethNums*2)).cuda().float()
        # self.att_mask = get_subsequent_mask(self.inputT)
        # self.att_mask = self.att_mask.view(1, 1, cfg.TeethNums*2, cfg.TeethNums*2)
        # self.att_mask[0, 0, :cfg.TeethNums, :cfg.TeethNums] = True