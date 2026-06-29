import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConv(nn.Module):
    """
    多尺度卷积模块
    输入: x [batch, seq_len, embed_dim]
    输出: [batch, seq_len, out_dim_total]
    """
    def __init__(self, in_dim, out_dim, kernel_sizes=[1,3,5], dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_dim, out_dim, kernel_size=k, padding=(k-1)//2, bias=False)
            for k in kernel_sizes
        ])
        self.dropout = dropout

    def forward(self, x, training = None):
        # x: [batch, seq_len, embed_dim] -> [batch, embed_dim, seq_len]
        # x = x.transpose(1,2)
        out = [conv(x) for conv in self.convs]  # 每个卷积 [batch, out_dim, seq_len]
        out = torch.cat(out, dim=1)  # [batch, out_dim_total, seq_len]
        out = F.dropout(out, p=self.dropout, training=training)
        # out = out.transpose(1,2)  # [batch, seq_len, out_dim_total]
        return out

# class MultiScaleAttention(nn.Module):
#     """
#     多尺度 Transformer 注意力
#     输入: x [seq_len, batch, embed_dim]
#     输出: [seq_len, batch, embed_dim]
#     """
#     def __init__(self, embed_dim, num_heads=8, dropout=0.1):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
#         self.norm = nn.LayerNorm(embed_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x: [seq_len, batch, embed_dim]
#         attn_out, _ = self.attn(x, x, x)
#         x = x + self.dropout(attn_out)  # 残差连接
#         x = self.norm(x)
#         return x

# # 测试运行
# if __name__ == "__main__":
#     batch, seq_len, embed_dim = 16, 30, 768
#     x = torch.randn(batch, seq_len, embed_dim)

#     # 多尺度卷积
#     conv_module = MultiScaleConv(in_dim=embed_dim, out_dim=256, kernel_sizes=[1,3,5], dropout=0.1)
#     x_conv = conv_module(x)
#     print("MultiScaleConv output:", x_conv.shape)  # [16, 30, 768] if 256*3=768

#     # 多尺度注意力
#     attn_module = MultiScaleAttention(embed_dim=768, num_heads=8, dropout=0.1)
#     x_attn = x_conv.transpose(0,1)  # Transformer要求 [seq_len, batch, embed_dim]
#     x_attn = attn_module(x_attn)
#     print("MultiScaleAttention output:", x_attn.shape)  # [30, 16, 768]
