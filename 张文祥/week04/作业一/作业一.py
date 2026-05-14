import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Transformer层实现
包含：
1. 多头注意力机制 (Multi-Head Attention)
2. 前馈神经网络 (Feed Forward Network)
3. 残差连接和层归一化
"""

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, n_heads, seq_len, d_k]
            K: [batch_size, n_heads, seq_len, d_k]
            V: [batch_size, n_heads, seq_len, d_v]
            mask: [batch_size, 1, seq_len, seq_len] 可选
        
        Returns:
            output: [batch_size, n_heads, seq_len, d_v]
            attn_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        d_k = Q.size(-1)
        
        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用mask（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads):
        """
        Args:
            d_model: 模型维度
            n_heads: 头数
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 确保d_model能被n_heads整除
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # 线性变换层
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # 输出层
        self.W_O = nn.Linear(d_model, d_model)
        
        # 缩放点积注意力
        self.attention = ScaledDotProductAttention()
    
    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, seq_len, d_model]
            K: [batch_size, seq_len, d_model]
            V: [batch_size, seq_len, d_model]
            mask: 可选
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attn_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        
        # 线性变换并分割成多头
        Q = self.W_Q(Q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        output, attn_weights = self.attention(Q, K, V, mask)
        
        # 拼接多头结果
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出线性变换
        output = self.W_O(output)
        
        return output, attn_weights


class PositionWiseFeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    """Transformer层（编码器层）"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            n_heads: 多头注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        
        # 多头注意力
        self.multi_head_attn = MultiHeadAttention(d_model, n_heads)
        
        # 前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 注意力掩码（可选）
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attn_weights: [batch_size, n_heads, seq_len, seq_len]
        """
        # 多头注意力 + 残差连接 + 层归一化
        attn_output, attn_weights = self.multi_head_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x, attn_weights


# 测试代码
if __name__ == "__main__":
    # 超参数设置
    batch_size = 2
    seq_len = 5
    d_model = 64
    n_heads = 4
    d_ff = 128
    
    # 创建Transformer层
    transformer_layer = TransformerLayer(d_model, n_heads, d_ff)
    print("Transformer层创建成功")
    print(f"模型参数数量: {sum(p.numel() for p in transformer_layer.parameters())}")
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    output, attn_weights = transformer_layer(x)
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    # 验证输出维度
    assert output.shape == x.shape, "输出维度错误"
    assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len), "注意力权重维度错误"
    
    print("\n✓ Transformer层测试通过！")
