import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
基于PyTorch框架的文本多分类任务
任务描述：对包含"你"字的5字文本，"你"在第几位就属于第几类（1-5分类）
示例：
- "你好世界呀" -> "你"在第1位 -> 第1类
- "爱你每一天" -> "你"在第2位 -> 第2类
- "我爱你中国" -> "你"在第3位 -> 第3类
"""

# 中文字符集（常用汉字）
CHARS = "你我他她它这那上下左右前后天地山水日月星辰风雨云雪春夏秋冬" \
        "家国人民心思言语行走站立坐卧吃喝看听读写学习工作生活" \
        "美好快乐幸福平安健康成功梦想希望勇气智慧善良真诚" \
        "一二三四五六七八九十百千万亿零"

# 字符到索引的映射
char_to_idx = {char: idx + 1 for idx, char in enumerate(CHARS)}  # 0表示padding
idx_to_char = {idx + 1: char for idx, char in enumerate(CHARS)}

# 生成一个样本
def build_sample():
    # 随机选择"你"的位置（1-5）
    you_pos = random.randint(1, 5)
    
    # 生成5字文本
    text = []
    for i in range(5):
        if i + 1 == you_pos:
            text.append("你")
        else:
            # 随机选择一个其他字符
            other_char = random.choice([c for c in CHARS if c != "你"])
            text.append(other_char)
    
    # 将文本转换为索引序列
    text_idx = [char_to_idx.get(c, 0) for c in text]
    
    return text_idx, you_pos

# 生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

# RNN模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        _, h_n = self.rnn(x)   # h_n: (1, batch_size, hidden_size)
        out = self.fc(h_n.squeeze(0))  # (batch_size, num_classes)
        
        if y is not None:
            y = y - 1  # 将标签从1-5转换为0-4
            return self.loss(out, y)
        else:
            return out

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: (1, batch_size, hidden_size)
        out = self.fc(h_n.squeeze(0))  # (batch_size, num_classes)
        
        if y is not None:
            y = y - 1  # 将标签从1-5转换为0-4
            return self.loss(out, y)
        else:
            return out

# 评估函数
def evaluate(model, vocab_size, embedding_dim, hidden_size, num_classes):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    print("测试集各类样本数:", end=" ")
    for i in range(1, 6):
        count = (y == i).sum().item()
        print(f"类别{i}:{count}", end=" ")
    print()
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        pred_classes = torch.argmax(y_pred, dim=1)
        for pred, true in zip(pred_classes, y):
            if pred == (true - 1):
                correct += 1
            else:
                wrong += 1
    
    acc = correct / (correct + wrong)
    print(f"正确预测: {correct}, 错误: {wrong}, 正确率: {acc:.4f}")
    return acc

# 主训练函数
def main(model_type='lstm'):
    # 配置参数
    epoch_num = 30
    batch_size = 64
    train_sample = 5000
    seq_len = 5
    vocab_size = len(CHARS) + 1  # +1 for padding
    embedding_dim = 32
    hidden_size = 64
    num_classes = 5
    learning_rate = 0.001
    
    # 创建模型
    if model_type == 'rnn':
        model = RNNModel(vocab_size, embedding_dim, hidden_size, num_classes)
        model_name = "RNN"
    else:
        model = LSTMModel(vocab_size, embedding_dim, hidden_size, num_classes)
        model_name = "LSTM"
    
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 生成训练集
    train_x, train_y = build_dataset(train_sample)
    
    # 训练过程
    print(f"===== {model_name}模型训练 =====")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        
        avg_loss = np.mean(watch_loss)
        print(f"第{epoch+1}轮 | 平均loss: {avg_loss:.4f}", end=" ")
        
        acc = evaluate(model, vocab_size, embedding_dim, hidden_size, num_classes)
        log.append([acc, avg_loss])
    
    # 保存模型
    torch.save(model.state_dict(), f"{model_name.lower()}_model.bin")
    print(f"\n模型已保存为: {model_name.lower()}_model.bin")
    
    # 画图
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(range(len(log)), [l[1] for l in log], label='loss', color='orange')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model

# 预测函数
def predict(model_path, input_texts, model_type='lstm'):
    vocab_size = len(CHARS) + 1
    embedding_dim = 32
    hidden_size = 64
    num_classes = 5
    
    # 加载模型
    if model_type == 'rnn':
        model = RNNModel(vocab_size, embedding_dim, hidden_size, num_classes)
    else:
        model = LSTMModel(vocab_size, embedding_dim, hidden_size, num_classes)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 处理输入文本
    X = []
    for text in input_texts:
        text_idx = [char_to_idx.get(c, 0) for c in text]
        X.append(text_idx)
    
    X = torch.LongTensor(X)
    
    with torch.no_grad():
        y_pred = model(X)
        pred_classes = torch.argmax(y_pred, dim=1)
    
    # 输出结果
    print("\n预测结果:")
    for text, pred in zip(input_texts, pred_classes):
        true_pos = text.index("你") + 1 if "你" in text else 0
        pred_pos = pred + 1
        print(f"文本: {text} | '你'实际位置: {true_pos} | 预测位置: {pred_pos}")

if __name__ == "__main__":
    # 训练LSTM模型
    model = main(model_type='lstm')
    
    # 测试示例
    test_texts = [
        "你好世界呀",  # "你"在第1位
        "爱你每一天",  # "你"在第2位
        "我爱你中国",  # "你"在第3位
        "世界有你美",  # "你"在第4位
        "明天想见你",  # "你"在第5位
        "你是我的梦",  # "你"在第1位
        "心中有你在",  # "你"在第3位
    ]
    
    predict("lstm_model.bin", test_texts)
