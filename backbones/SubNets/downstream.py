from torch import nn
import torch
class WindowAttentionDownsampler(nn.Module):
    def __init__(self, input_dim, target_dim, output_len):
        super().__init__()
        # 1. 维度投影层：如果输入维度不是 768，先映射到 768
        self.proj = nn.Linear(input_dim, target_dim) if input_dim != target_dim else nn.Identity()

        # 2. 局部特征提取：使用深度卷积（Depthwise Conv）或普通卷积
        # 卷积可以捕捉到语音/视频在时间轴上的局部变化
        self.conv = nn.Conv1d(target_dim, target_dim, kernel_size=3, stride=1, padding=1)

        # 3. 自适应对齐：强制将 L 维度压缩到 30
        # 混合池化：同时保留平均特征和显著特征（如语音中的尖锐信号）
        self.avg_pool = nn.AdaptiveAvgPool1d(output_len)
        self.max_pool = nn.AdaptiveMaxPool1d(output_len)

        self.norm = nn.LayerNorm(target_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: [B, L, input_dim]

        # 步骤 1: 维度映射 -> [B, L, 768]
        x = self.proj(x)

        # 步骤 2: 准备进入 Conv1d -> [B, 768, L]
        x = x.transpose(1, 2)

        # 步骤 3: 局部特征提取
        x = torch.relu(self.conv(x))

        # 步骤 4: 自适应对齐至 output_len (30)
        # 结合平均值和最大值，防止信息丢失
        x = (self.avg_pool(x) + self.max_pool(x)) / 2

        # 步骤 5: 还原形状 -> [B, 30, 768]
        x = x.transpose(1, 2)
        return self.norm(self.dropout(x))


import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F


class ToMCrossAttention(nn.Module):
    """
    ToM-Gated Cross Modal Attention
    - text_feat: [B, Lt, D]
    - other_feat: [B, Lm, D]  (如 Video/Audio)
    - ToM state 在 __init__ 初始化（可学习）
    """

    def __init__(self, dim, tom_dim=8, num_heads=8):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # =============================
        # 1. 多头投影
        # =============================
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # =============================
        # 2. ★ ToM 参数体系（科学8维）
        #    可学习的心理因素向量
        # =============================
        # 8维心理学解释：
        #  0 text_confidence
        #  1 audio_confidence
        #  2 video_confidence
        #  3 semantic_clarity
        #  4 temporal_stability
        #  5 emotion_intensity
        #  6 noise_level
        #  7 context_coherence
        self.tom_state = nn.Parameter(torch.randn(1, tom_dim))

        # =============================
        # 3. ToM gate MLP
        # =============================
        self.gate_mlp = nn.Sequential(
            nn.Linear(tom_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, text_feat, other_feat, return_att=False):
        """
        return:
            out: 融合后特征 [B, Lt, D]
            gate: ToM 门控 [B, H, 1, d]
            att: 注意力图 [B, H, Lt, Lm]
        """

        B, Lt, D = text_feat.shape
        B, Lm, _ = other_feat.shape

        # =============================
        # 1. QKV 投影
        # =============================
        Q = self.q_proj(text_feat)      # [B, Lt, D]
        K = self.k_proj(other_feat)     # [B, Lm, D]
        V = self.v_proj(other_feat)     # [B, Lm, D]

        # 多头展开
        Q = Q.view(B, Lt, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lm, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lm, self.num_heads, self.head_dim).transpose(1, 2)

        # =============================
        # 2. ToM Gate
        # =============================
        # 扩展到 batch
        tom_state = self.tom_state.expand(B, -1)

        gate = self.gate_mlp(tom_state)              # [B, D]
        gate = gate.view(B, self.num_heads, 1, self.head_dim)

        # 对 K/V 做门控
        K = K * gate
        V = V * gate

        # =============================
        # 3. Multi-Head Attention
        # =============================
        att = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.softmax(dim=-1)

        out = att @ V                  # [B, H, Lt, d]
        out = out.transpose(1, 2).reshape(B, Lt, D)
        out = self.out_proj(out)

        if return_att:
            return out, gate, att

        return out