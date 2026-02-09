import math

import torch
from torch import nn

# 时间嵌入模块 time embedding module
class TimeEmbedding(nn.Module):
    def __init__(
        self,
        time_channels: int,
    ):
        super().__init__()
        # time_channels == model_channels * 4
        self.time_channels = time_channels
        
        self.linear_1 = nn.Linear(time_channels // 4, time_channels)
        self.act_1 = nn.SiLU()
        self.linear_2 = nn.Linear(time_channels, time_channels)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # d == model_channels //2 == time_channels // 8
        d = self.time_channels //8
        freq = torch.exp(-math.log(10000)/ (d-1) * torch.arange(d, device=t.device))
        # t.shape == (batch_size, )， freq.shape == (d, )
        # emb.shape == (batch_size, d)
        emb = t[:, None] * freq[None, :]
        
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.linear_2(self.act_1(self.linear_1(emb)))
       
       
#残差块 residual module
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int, 
        time_channels: int, 
        num_group: int = 32, 
        drop_out: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.num_group = num_group
        self.drop_out = drop_out
        
        # 组归一化，激活，卷积
        self.norm_1 = nn.GroupNorm(self.num_group, in_channels)
        self.act_1 = nn.SiLU()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 时间激活，时间嵌入
        self.time_act = nn.SiLU()
        self.time_emb_proj = nn.Linear(time_channels, out_channels)
        
        # 组归一化，激活，drop out，卷积
        self.norm_2 = nn.GroupNorm(self.num_group, out_channels)
        self.act_2 = nn.SiLU()
        self.dropout = nn.Dropout(self.drop_out)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 残差链接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # h代表隐藏状态，中间结果
        h = self.conv_1(self.act_1(self.norm_1(x)))
        # t.shape == (batch_size, time_channels) 时间嵌入的shape
        # x.shape == (batch_size, out_channels, height, width) 图像的shape
        h += self.time_emb_proj(self.time_act(t))[:, :, None, None]
        h = self.conv_2(self.dropout(self.act_2(self.norm_2(h))))
        
        return h + self.shortcut(x)
    
    
# 多头自注意模块 Multi-headed self-attention module
class AttentionBlock(nn.Module):
    def __init__(
        self, 
        out_channels: int,
        num_heads: int = 1,
        d_k: int | None = None, #每一个头对应的qkv的向量长度
        num_group: int =32
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        if d_k is None:
            assert out_channels % num_heads == 0, f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads})"
            d_k = out_channels // num_heads
        self.d_k = d_k
        self.num_group = num_group
        
        self.norm = nn.GroupNorm(self.num_group, out_channels)
        self.input_proj = nn.Linear(out_channels, num_heads * d_k * 3)
        self.output = nn.Linear(num_heads * d_k, out_channels)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        
        x = self.norm(x)
        seq_len = h * w
        # (b, c, h, w) -> (b, c, seq_len)
        x = x.reshape(b, c, seq_len)
        
        # (b, c, seq_len) -> (b, seq_len, c)
        x = x.permute(0, 2, 1)
        
        # (b, seq_len, c) -> (b, seq_len, num_heads * d_k * 3)
        qkv: torch.Tensor = self.input_proj(x)
        
        # (b, seq_len, num_heads * d_k * 3) -> (b, num_heads, seq_len, d_k * 3)
        qkv = qkv.reshape(b, seq_len, self.num_heads, self.d_k * 3)
        
        # (b, num_heads, seq_len, d_k * 3) -> (b, num_heads, seq_len, d_k) * 3
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        self.scale = self.d_k ** -0.5
        # (b, seq_len, num_heads, d_k),(b, seq_len, num_heads, d_k) -> (b, seq_len, seq_len, num_heads)
        attention_scores: torch.Tensor = torch.einsum("bihd,bjhd -> bijh", q, k) * self.scale
        
        # (b, seq_len, seq_len, num_heads) -> (b, seq_len, seq_len, num_heads)
        attention_weights = attention_scores.softmax(dim=2)
        
        # (b, seq_len, seq_len, num_heads), (b, seq_len, num_heads, d_k) -> (b, seq_len, num_heads, d_k)
        attention_aggregation = torch.einsum("bijh,bjhd -> bihd", attention_weights, v)
        
        # (b, seq_len, num_heads, d_k) -> (b, seq_len, num_heads, d_k)
        attention_aggregation = attention_aggregation.reshape(b, seq_len, -1)
        
        # (b, seq_len, num_heads, d_k) -> (b, seq_len, out_channels)
        res: torch.Tensor = self.output(attention_aggregation)
        
        # (b, seq_len, out_channels) -> (b, c, h, w)
        res = res.permute(0, 2, 1).reshape(b, c, h, w)
        
        return res + x_in
    

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int, 
        time_channels: int, 
        is_attn: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.is_attn = is_attn
        
        self.res_block = ResidualBlock(in_channels, out_channels, time_channels)
        if is_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
            
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int, 
        time_channels: int, 
        is_attn: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.is_attn = is_attn
        
        self.res_block = ResidualBlock(in_channels, out_channels, time_channels)
        if is_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
            
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x, t)
        x = self.attn(x)
        return x
    

class DownSample(nn.Module):
    def __init__(
        self,
        out_channels: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        # H_new = (H_old + 2 * padding - kernel_size) // stride +1
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, t:torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)
    

class UpSample(nn.Module):
    def __init__(
        self,
        out_channels: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        # H_new = (H_old - 1) * stride -2 * padding + kernel_size + output_padding
        self.transpose_conv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, t:torch.Tensor | None = None) -> torch.Tensor:
        return self.transpose_conv(x)
    

class MiddleBlock(nn.Module):
    def __init__(
        self,
        out_channels: int,
        time_channels: int,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.time_channels = time_channels
        
        self.res_1 = ResidualBlock(out_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels)
        self.res_2 = ResidualBlock(out_channels, out_channels, time_channels)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res_1(x, t)
        x = self.attn(x)
        x = self.res_2(x, t)
        return x
    

class DDPMUnet(nn.Module):
    def __init__(
        self,
        img_channels: int, 
        model_channels: int,
        channel_mults: list[int] = [1, 2, 2, 2],
        is_attn: list[bool] = [False, False, False, False],
        num_res_block: int = 2, 
        num_groups : int = 32,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.model_channels = model_channels
        self.channel_mults = channel_mults
        self.is_attn = is_attn
        self.num_res_block = num_res_block
        self.num_groups = num_groups
        self.time_channels = model_channels * 4
        
        # 输入图像映射
        self.img_proj = nn.Conv2d(img_channels, model_channels, kernel_size=3, padding=1)
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(self.time_channels)
        
        # 通道增加与下采样部分
        self.down_blocks_num = len(self.channel_mults)
        self.down_blcoks = nn.ModuleList()
        cur_channels = model_channels
        self.channel_stack = [cur_channels]
        for i in range(self.down_blocks_num):
            nxt_channels = cur_channels * self.channel_mults[i]
            
            for _ in range(self.num_res_block):
                self.down_blcoks.append(DownBlock(cur_channels, nxt_channels, self.time_channels, self.is_attn[i]))
                cur_channels = nxt_channels
                self.channel_stack.append(cur_channels)
                
            if i < self.down_blocks_num - 1:
                self.down_blcoks.append(DownSample(cur_channels))
                
        self.middle_block = MiddleBlock(cur_channels, self.time_channels)
        
        # 通道减少与上采样部分
        self.up_blocks = nn.ModuleList()
        # Decoder pathway starts with middle block output channels
        decoder_channels = cur_channels
        for i in range(self.down_blocks_num -1, -1, -1):
            # Add UpSample BEFORE the blocks (so upsample runs before concat+block in forward)
            if i < self.down_blocks_num - 1:
                self.up_blocks.append(UpSample(decoder_channels))

            for _ in range(self.num_res_block):
                skip_channels = self.channel_stack.pop()
                nxt_channels = self.channel_stack[-1]
                # Input: skip channels + decoder pathway channels
                # Output: nxt_channels (target output channels)
                self.up_blocks.append(UpBlock(skip_channels + decoder_channels, nxt_channels, self.time_channels, self.is_attn[i]))
                # After UpBlock, decoder pathway channels become the output channels
                decoder_channels = nxt_channels
                
        # 输出部分
        self.norm = nn.GroupNorm(self.num_groups, model_channels)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(model_channels, img_channels, kernel_size=3, padding=1)
        
                       
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Store original input size for final resize
        orig_size = x.shape[2:]

        x = self.img_proj(x)
        t = self.time_embedding(t)

        h = [x]
        for down_block in self.down_blcoks:
            if isinstance(down_block, DownSample):
                x = down_block(x, t)
            else:
                x = down_block(x, t)
                h.append(x)

        x = self.middle_block(x, t)
        
        for up_block in self.up_blocks:
            if isinstance(up_block, UpSample):
                x = up_block(x, t)
            else:
                s = h.pop()
                # Resize skip connection to match decoder spatial dimensions if needed
                if s.shape[2:] != x.shape[2:]:
                    s = torch.nn.functional.interpolate(s, size=x.shape[2:], mode='nearest')
                x = torch.cat((s, x), dim=1)
                x = up_block(x, t) 
                
        x = self.conv(self.act(self.norm(x)))
        # Resize output to match original input size
        if x.shape[2:] != orig_size:
            x = torch.nn.functional.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x