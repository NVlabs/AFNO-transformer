from torch import nn
import torch
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

class AttentionLS(nn.Module):
    """Implementation for long-short term attention.
    Flexible options for using window attention, global token and dynamic projection.
    Args:
        dim: input and output feature dimension.
        num_heads: number of attention heads.
        qkv_bias: whether to use bias for the projection of query, key and values.
        qk_scale: scale factor on query and key for numerical stability.
                  By default, set to square root of head dimensions.
        attn_drop: dropout probability for attention matrix.
        proj_drop: dropout probability for the final output.
        rpe: whether to use relative position encoding.
        nglo: number of global tokens (e.g., CLS).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., rpe=False, nglo=1,
                 dp_rank=2, w=2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.nglo = nglo

        # Equals to segment size (w) in the paper.
        self.window_size = w
        # Equals to r in the paper.
        self.dp_rank = dp_rank

        if self.dp_rank > 0:
            self.to_dynamic_projection = nn.Linear(dim, dp_rank * num_heads)
        # The LN of DualLN corresponding to dynamic projection
        self.dual_ln_dp = nn.LayerNorm(dim)
        # The LN of DualLN corresponding to all the tokens
        self.dual_ln_full = nn.LayerNorm(dim)

        # Adapted from ViL: https://github.com/microsoft/vision-longformer/blob/main/src/models/layers/longformer2d.py#L55-L100
        # We only add RPE to window attention.
        # Unnecessary to add bias for global tokens, since DualLN already adds biases.
        self.rpe = rpe
        if rpe:
            # handle the boarder conditions...
            w_pad = int(w*0.5)
            self.local_relative_position_bias_table = nn.Parameter(
                torch.zeros(2 * (w + w_pad - 1) * (2 * w_pad + w + 1) + 1, num_heads))
            trunc_normal_(self.local_relative_position_bias_table, std=.02)

            # get pair-wise relative position index
            coords_h = torch.arange(-w_pad, w_pad + w)
            coords_w = torch.arange(-w_pad, w_pad + w)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, 2w, 2w
            coords = coords.view(2, (w + w_pad * 2)**2).transpose(0, 1).unsqueeze(0) # 1, 4w**2, 2
            q_coords_hw = torch.arange(0, w)
            q_coords = torch.stack(torch.meshgrid([q_coords_hw, q_coords_hw])) # 2, w, w
            q_coords = q_coords.view(2, w**2).transpose(0, 1).unsqueeze(1) # w**2, 1, 2
            relative_coords = q_coords - coords
            relative_coords += w_pad + w - 1  # shift to start from 0
            relative_coords[:, :, 0] *= 2 * w_pad + w
            relative_position_index = relative_coords.sum(-1)  # w^2, 4w^2
            self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, nx=None, ny=None):
        B, N, C = x.shape
        N_feat = N - self.nglo
        self.img_size = nx
        qkv = self.qkv(x)
        # query, key, value
        q, k, v = qkv.chunk(3, dim=2)
        q = q.mul(self.scale)

        # Layer norm on the projected keys and values
        k = self.dual_ln_full(k)
        v = self.dual_ln_full(v)

        # output size: bsz x n_heads x seqlen x d
        if self.nglo > 0:
            q_cls, q = q[:, :self.nglo], q[:, self.nglo:]
            k_cls, k = k[:, :self.nglo], k[:, self.nglo:]
            v_cls, v = v[:, :self.nglo], v[:, self.nglo:]

            q_cls = q_cls.reshape(B, self.nglo, self.num_heads, C // self.num_heads).transpose(1, 2)
            k_cls = k_cls.reshape(B, self.nglo, self.num_heads, C // self.num_heads).transpose(1, 2)
            v_cls = v_cls.reshape(B, self.nglo, self.num_heads, C // self.num_heads).transpose(1, 2)

        q = q.reshape(B, N_feat, self.num_heads, C//self.num_heads).transpose(1, 2)
        k = k.reshape(B, N_feat, self.num_heads, C//self.num_heads).transpose(1, 2)
        v = v.reshape(B, N_feat, self.num_heads, C//self.num_heads).transpose(1, 2)

        # Long-range Attention (Dynamic Projection)
        if self.dp_rank > 0:
            # b x h x r x (l w)
            # Compute the projection matrix (P_i in the paper)
            c_scores = self.to_dynamic_projection(x[:, self.nglo:]).transpose(1, 2).contiguous().view(
                B, self.num_heads, self.dp_rank, -1)
            c_scores = c_scores.softmax(dim=-1, dtype=torch.float32).to(x)
            # b x h x r x d
            k_lms = c_scores.matmul(k)
            k_lms = k_lms.transpose(1, 2).contiguous().view(B, self.dp_rank, -1)
            k_lms = self.dual_ln_dp(k_lms).view(B, self.dp_rank, self.num_heads, -1).contiguous().permute(0, 2, 3, 1)
            # b x h x (lw) x r
            dots_all = q.matmul(k_lms)

            if self.window_size > 0:
                # Switch the order of dimensions if using window attention.
                dots_all = self.group_dots(dots_all)
        else:
            dots_all = None

        # Short-term Attention (Window Attention)
        # In our window attention, each token attends to at most (4w^2) tokens.
        if self.window_size > 0:
            dots_win = self.compute_window_scores(q, k)
            w2 = int(self.window_size*self.window_size)

            if self.rpe:
                w_pad = int(0.5 * self.window_size)
                local_relative_position_bias = self.local_relative_position_bias_table[
                    self.relative_position_index.view(-1)].view(1, w2, (w_pad*2 + self.window_size)**2, -1)  # w^2, kv_nums,H
                local_relative_position_bias = local_relative_position_bias.permute(
                    0, 3, 1, 2).expand(B, -1, -1, -1).unsqueeze(2).unsqueeze(2)

                dots_win += local_relative_position_bias
            if dots_all is None:
                dots_all = dots_win
            else:
                dots_all = torch.cat([dots_all, dots_win], dim=-1)

        # Global token.
        if self.nglo > 0:
            # and compute the scores of queries on CLS
            dots_q_cls = q.matmul(k_cls.transpose(-1, -2))

            if self.window_size > 0:
                dots_q_cls = self.group_dots(dots_q_cls)
            dots_all = torch.cat([dots_all, dots_q_cls], dim=-1)

        attn = dots_all.softmax(dim=-1, dtype=torch.float32).to(x)
        attn = self.attn_drop(attn)
        out = 0
        if self.window_size > 0:
            offset = max(0, self.dp_rank)
            kv_group_size = self.window_size
            total_win_size = max(1, self.window_size // 2) * 2 + kv_group_size
            attn_win = attn[:, :, :, :, :, offset:offset + total_win_size ** 2]
            out += self.compute_window_pv(attn_win, v)
            attn = self.ungroup_dots(attn)

        # attn will be b x h x lw x n_k from now on
        if self.dp_rank > 0:
            attn_lm = attn[:, :, :, :self.dp_rank]
            v_lms = c_scores.matmul(v.float()).to(v).transpose(1, 2).contiguous().view(B, self.dp_rank, -1)
            v_lms = self.dual_ln_dp(v_lms).view(B, self.dp_rank, self.num_heads, -1).contiguous().transpose(1, 2)

            out += attn_lm.matmul(v_lms)

        if self.nglo > 0:
            attn_cls = attn[:, :, :, -self.nglo:]
            out += attn_cls.mul(v_cls)

            # b x h x 1 x lw
            cls_inner = q_cls.matmul(k_cls.transpose(-1, -2))
            cls_dots = q_cls.matmul(out.transpose(-1, -2))
            cls_dots = torch.cat([cls_inner, cls_dots], dim=-1)

            cls_dots = cls_dots.softmax(dim=-1, dtype=torch.float32).to(x)
            cls_next = cls_dots[:, :, :, self.nglo:].matmul(out) # the post_cls variant
            cls_next += cls_dots[:, :, :, :self.nglo].matmul(v_cls)

            out = torch.cat([cls_next, out], dim=2)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def compute_window_scores(self, q, k):
        """Compute the inner products for the window attention.
        Frist, divide the query into non-overlapping windows.
        Then, use torch.as_trided (implemented in self.get_overlapping_tiles) to create a view of the keys
        that corresponds to the windows with at most 2x memory overhead.
        Finally, compute the inner product.
        """
        # q: b h (l w) d
        b, h, _, d = q.shape
        side_size = max(self.window_size//2, 1)
        # q_group_size: segment size
        kv_width = 2 * side_size + self.window_size # assuming q_stride=1
        q_n_group = self.img_size // self.window_size
        q_tiles = q.reshape(b, h, q_n_group, self.window_size, q_n_group, self.window_size, d).permute(
            0, 1, 2, 4, 3, 5, 6)
        # q_tiles: b x h x n_group x n_group x w^2 x d
        q_tiles = q_tiles.contiguous().view(b, h, q_n_group, q_n_group, -1, d)

        # k_tiles: b x h x n_group x n_group x 9w^2 x d
        k_tiles = self.get_overlapping_tiles(k).contiguous().view(b, h, q_n_group, q_n_group, -1, d)
        # dot_tiles: b x h x n_group x n_group x w^2 x 9w^2
        dot_tiles = q_tiles.matmul(k_tiles.transpose(-1, -2))

        # fill "-inf" into the zero-padding parts
        dot_tiles = dot_tiles.view(b, h, q_n_group, q_n_group, -1, kv_width, kv_width)

        dot_tiles[:, :, 0, :, :, :side_size].fill_(float('-inf'))
        dot_tiles[:, :, -1, :, :, -side_size:].fill_(float('-inf'))
        dot_tiles[:, :, :, 0, :, :, :side_size].fill_(float('-inf'))
        dot_tiles[:, :, :, -1, :, :, -side_size:].fill_(float('-inf'))

        dot_tiles = dot_tiles.view(b, h, q_n_group, q_n_group, -1, kv_width ** 2)
        return dot_tiles

    def get_overlapping_tiles(self, x):
        """Get overlapping tiles in the 2D spatial domain, ensuring each query computes correlation with all neighbors
        """
        # x: b h (l w) d
        b, h, _, d = x.shape
        side_size = max(self.window_size // 2, 1)
        total_size = 2 * side_size + self.window_size
        kv_group_size = self.window_size
        kv_width = self.img_size

        x = x.view(b, h, kv_width, kv_width, d)
        x = F.pad(x, [0, 0, side_size, side_size, side_size, side_size], value=0)

        out_shape = [b, h, kv_width // kv_group_size, kv_width // kv_group_size,
                     total_size, total_size, d]
        in_stride = x.stride()
        out_stride = [in_stride[0], in_stride[1], in_stride[2] * kv_group_size, in_stride[3] * kv_group_size,
                      in_stride[2], in_stride[3], in_stride[4]]

        # note we ignored the boundary here
        return x.as_strided(size=out_shape, stride=out_stride)

    def compute_window_pv(self, attn, v):
        """Compute the inner product of attention matrix and the values for the window attention.
        """
        b, h, n_group, _, w2, n_k = attn.shape
        d = v.shape[-1]
        v_tiles = self.get_overlapping_tiles(v).contiguous().view(b, h, n_group, n_group, -1, d)

        # b x h x n_group x n_group x w^2 x d
        pv = attn.matmul(v_tiles)
        # return: b x h x (lw) x d
        ret = self.ungroup_dots(pv)

        return ret

    def group_dots(self, dots):
        b, h = dots.shape[:2]
        n_group = self.img_size // self.window_size
        dots = dots.reshape(b, h, n_group, self.window_size, n_group, self.window_size,
                            -1).permute(0, 1, 2, 4, 3, 5, 6)
        dots = dots.contiguous().view(b, h, n_group, n_group, self.window_size * self.window_size, -1)
        return dots

    def ungroup_dots(self, dots):
        b, h, n_group, _, _, n_keys = dots.shape
        dots = dots.reshape(b, h, n_group, n_group, self.window_size, self.window_size,
                            -1).permute(0, 1, 2, 4, 3, 5, 6)
        dots = dots.contiguous().view(b, h, -1, n_keys)
        return dots