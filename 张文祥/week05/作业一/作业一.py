import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os

# ==================== 位置编码 ====================
class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [seq_len, batch_size, d_model]"""
        return x + self.pe[:x.size(0)]

# ==================== 多头注意力 ====================
class MultiHeadAttention(nn.Module):
    """多头注意力机制（支持因果掩码）"""
    def __init__(self, hidden, n_head, dropout=0.1):
        super().__init__()
        assert hidden % n_head == 0
        
        self.n_head = n_head
        self.d_k = hidden // n_head
        
        # 一次性计算Q、K、V
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        B, T, H = x.shape
        
        # 计算Q、K、V
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # 各 [B, T, H]
        
        # 分割成多头
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)  # [B, n_head, T, d_k]
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, n_head, T, T]
        
        # 应用掩码（因果掩码或padding掩码）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        output = torch.matmul(attn, v)  # [B, n_head, T, d_k]
        
        # 拼接多头
        output = output.transpose(1, 2).contiguous().view(B, T, H)
        
        # 输出变换
        output = self.out(output)
        
        return output, attn

# ==================== 前馈网络 ====================
class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, hidden, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden, d_ff)
        self.fc2 = nn.Linear(d_ff, hidden)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==================== Transformer解码器层 ====================
class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, hidden, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden, n_head, dropout)
        self.feed_forward = FeedForward(hidden, d_ff, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接
        residual = x
        x, attn = self.self_attn(x, mask)
        x = self.dropout1(x)
        x = self.norm1(residual + x)
        
        # 前馈网络 + 残差连接
        residual = x
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = self.norm2(residual + x)
        
        return x, attn

# ==================== GPT风格语言模型 ====================
class GPTModel(nn.Module):
    """GPT风格的单向语言模型"""
    def __init__(self, vocab_size, hidden=512, n_head=8, d_ff=2048, n_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.max_len = max_len
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, hidden)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden, max_len)
        
        # Transformer解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden, n_head, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(hidden, vocab_size)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] 输入序列
        
        Returns:
            logits: [batch_size, seq_len, vocab_size] 预测概率
        """
        B, T = x.shape
        
        # 词嵌入
        x = self.embedding(x)  # [B, T, hidden]
        
        # 添加位置编码（需要转置）
        x = x.transpose(0, 1)  # [T, B, hidden]
        x = self.pos_encoding(x)  # [T, B, hidden]
        x = x.transpose(0, 1)  # [B, T, hidden]
        
        # 生成因果掩码（上三角矩阵）
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        
        # 逐层计算
        for layer in self.layers:
            x, _ = layer(x, mask)
        
        # 输出层
        logits = self.fc_out(x)  # [B, T, vocab_size]
        
        return logits
    
    def generate(self, start_tokens, max_len=100, temperature=1.0, top_k=50):
        """
        文本生成
        Args:
            start_tokens: [batch_size, start_seq_len] 起始token序列
            max_len: 生成的最大长度
            temperature: 温度参数，控制随机性
            top_k: top-k采样
        
        Returns:
            generated: [batch_size, max_len] 生成的序列
        """
        self.eval()
        generated = start_tokens
        
        with torch.no_grad():
            for _ in range(max_len - start_tokens.size(1)):
                # 获取最后max_len个token（避免过长）
                inputs = generated[:, -self.max_len:]
                
                # 前向传播
                logits = self(inputs)  # [B, T, vocab_size]
                
                # 只取最后一个位置的预测
                logits = logits[:, -1, :] / temperature  # [B, vocab_size]
                
                # Top-k采样
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # 采样
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
                
                # 拼接
                generated = torch.cat([generated, next_token], dim=1)
        
        self.train()
        return generated

# ==================== 数据预处理 ====================
def build_vocab(text):
    """构建词汇表"""
    chars = sorted(list(set(text)))
    vocab = {char: idx for idx, char in enumerate(chars)}
    vocab['<PAD>'] = len(vocab)
    vocab['<UNK>'] = len(vocab)
    return vocab

def encode(text, vocab):
    """将文本编码为token序列"""
    return [vocab.get(char, vocab['<UNK>']) for char in text]

def decode(tokens, vocab):
    """将token序列解码为文本"""
    idx_to_char = {idx: char for char, idx in vocab.items()}
    return ''.join([idx_to_char.get(token, '<UNK>') for token in tokens])

def create_dataset(text, vocab, seq_len=64, stride=32):
    """创建训练数据集"""
    tokens = encode(text, vocab)
    data = []
    
    for i in range(0, len(tokens) - seq_len, stride):
        input_seq = tokens[i:i+seq_len]
        target_seq = tokens[i+1:i+seq_len+1]
        data.append((input_seq, target_seq))
    
    return data

# ==================== 训练函数 ====================
def train(model, dataset, vocab, epochs=10, batch_size=32, lr=1e-4, device='cuda'):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.get('<PAD>', -100))
    
    print(f"开始训练... 设备: {device}, 数据集大小: {len(dataset)}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 打乱数据
        random.shuffle(dataset)
        
        # 分批训练
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # 准备数据
            inputs = torch.tensor([item[0] for item in batch], dtype=torch.long).to(device)
            targets = torch.tensor([item[1] for item in batch], dtype=torch.long).to(device)
            
            # 前向传播
            logits = model(inputs)  # [B, T, vocab_size]
            
            # 计算损失
            loss = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 打印进度
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        # 每轮训练后生成一段文本
        if (epoch + 1) % 2 == 0:
            print("\n生成示例:")
            start_text = "今天天气"
            start_tokens = torch.tensor([encode(start_text, vocab)], dtype=torch.long).to(device)
            generated = model.generate(start_tokens, max_len=50)
            print(decode(generated[0].cpu().numpy(), vocab))
            print("-" * 50)
    
    return model

# ==================== 主函数 ====================
def main():
    # 使用中文文本作为训练数据
    text = """
    今天天气很好，我想出去走走。
    外面阳光明媚，鸟儿在树上唱歌。
    我来到公园，看到很多人在散步。
    有的在跑步，有的在打太极，还有的在聊天。
    公园里的花开得很漂亮，有红色的玫瑰，粉色的桃花，还有黄色的迎春花。
    微风吹过，花香扑鼻，让人感到心旷神怡。
    我找了一个长椅坐下，享受这美好的时光。
    时间过得真快，不知不觉已经到了下午。
    我决定回家，明天再来公园散步。
    生活中的小确幸，就是这样简单而美好。
    """ * 10  # 重复多次增加数据量
    
    # 构建词汇表
    vocab = build_vocab(text)
    print(f"词汇表大小: {len(vocab)}")
    print(f"词汇: {list(vocab.keys())[:20]}...")
    
    # 创建数据集
    dataset = create_dataset(text, vocab, seq_len=64, stride=32)
    print(f"训练样本数: {len(dataset)}")
    
    # 创建模型
    model = GPTModel(
        vocab_size=len(vocab),
        hidden=256,
        n_head=4,
        d_ff=1024,
        n_layers=4,
        max_len=128,
        dropout=0.1
    )
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 训练模型
    model = train(model, dataset, vocab, epochs=20, batch_size=8, lr=1e-4, device=device)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }, 'gpt_language_model.pth')
    print("\n模型已保存到 gpt_language_model.pth")
    
    # 生成文本测试
    print("\n=== 文本生成测试 ===")
    test_starts = ["今天天气", "公园里", "我喜欢", "春天来了"]
    for start in test_starts:
        start_tokens = torch.tensor([encode(start, vocab)], dtype=torch.long).to(device)
        generated = model.generate(start_tokens, max_len=60, temperature=0.8)
        print(f"输入: {start}")
        print(f"生成: {decode(generated[0].cpu().numpy(), vocab)}")
        print()

if __name__ == "__main__":
    main()
