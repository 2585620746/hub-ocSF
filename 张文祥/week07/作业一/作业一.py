"""
BERT 序列标注 NER 训练脚本
支持 JSON 格式数据集
"""

import os
import json
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "peoples_daily"  
MODEL_PATH = ROOT.parent / "week07" / "models" / "Qwen" / "Qwen2-0.5B-Instruct"  
OUTPUT_DIR = ROOT / "outputs"

# ────────────────────────────────────────────────────────────────────────────────
# 数据集类
# ────────────────────────────────────────────────────────────────────────────────
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        
        # 从实体转换为字符级标签
        char_labels = ["O"] * len(text)
        for ent in item.get("entities", []):
            start = ent["start"]
            end = ent.get("end", start + len(ent["text"]))
            ent_type = ent["type"]
            
            if start < len(text):
                char_labels[start] = f"B-{ent_type}"
                for i in range(start + 1, min(end, len(text))):
                    char_labels[i] = f"I-{ent_type}"

        # 处理子词切分
        input_ids = []
        label_ids = []
        
        for char, label in zip(text, char_labels):
            sub_tokens = self.tokenizer.tokenize(char)
            if not sub_tokens:
                sub_tokens = [self.tokenizer.unk_token]
            
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(sub_tokens))
            label_ids.append(self.label2id.get(label, self.label2id["O"]))
            label_ids.extend([-100] * (len(sub_tokens) - 1))

        # 截断和 padding
        input_ids = input_ids[:self.max_len]
        label_ids = label_ids[:self.max_len]
        
        padding_len = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        label_ids += [-100] * padding_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "attention_mask": torch.tensor([1] * (self.max_len - padding_len) + [0] * padding_len, dtype=torch.long)
        }

# ────────────────────────────────────────────────────────────────────────────────
# 模型类
# ────────────────────────────────────────────────────────────────────────────────
class BertForTokenClassification(nn.Module):
    def __init__(self, bert_path, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)

        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.shape[-1]), labels.view(-1)
            )
            return loss, logits
        return logits

# ────────────────────────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────────────────────────
def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def build_label_schema(label_names):
    """从 label_names.json 构建标签体系"""
    labels = ["O"]
    for name in label_names:
        labels.append(f"B-{name}")
        labels.append(f"I-{name}")
    
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label

def compute_f1(y_true, y_pred):
    """计算实体级 F1 分数（不依赖 seqeval）"""
    tp = 0
    fp = 0
    fn = 0
    
    for true_seq, pred_seq in zip(y_true, y_pred):
        # 提取实体
        true_entities = extract_entities(true_seq)
        pred_entities = extract_entities(pred_seq)
        
        # 计算 TP, FP, FN
        tp += len(true_entities & pred_entities)
        fp += len(pred_entities - true_entities)
        fn += len(true_entities - pred_entities)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1

def extract_entities(labels):
    """从标签序列中提取实体集合"""
    entities = set()
    current_entity = []
    
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current_entity:
                entities.add(tuple(current_entity))
            current_entity = [label[2:], i, i]  # (type, start, end)
        elif label.startswith("I-") and current_entity:
            current_entity[-1] = i  # 更新 end
        else:
            if current_entity:
                entities.add(tuple(current_entity))
                current_entity = []
    
    if current_entity:
        entities.add(tuple(current_entity))
    
    return entities

# ────────────────────────────────────────────────────────────────────────────────
# 评估函数
# ────────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, id2label, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
            for pred, label in zip(preds, labels):
                pred_seq = []
                label_seq = []
                for p, l in zip(pred, label):
                    if l != -100:
                        pred_seq.append(id2label[p.item()])
                        label_seq.append(id2label[l.item()])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)
    
    f1 = compute_f1(all_labels, all_preds)
    return f1, (all_labels, all_preds)

# ────────────────────────────────────────────────────────────────────────────────
# 参数解析
# ────────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="BERT NER 序列标注训练")
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--max_len", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# 主函数
# ────────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 路径设置
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print("\n加载数据...")
    train_data = load_json(data_dir / "train.json")
    val_data = load_json(data_dir / "validation.json")
    test_data = load_json(data_dir / "test.json")

    print(f"标签类型: {label_names}")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    print(f"测试集: {len(test_data)} 条")

    # ── 构建标签体系 ──────────────────────────────────────────────────────────
    label2id, id2label = build_label_schema(label_names)
    print(f"\n标签体系: {list(label2id.keys())}")

    # ── 加载 Tokenizer ────────────────────────────────────────────────────────
    print(f"\n加载 Tokenizer: {args.model_path}")
    tokenizer = BertTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 构建数据集 ────────────────────────────────────────────────────────────
    train_dataset = NERDataset(train_data, tokenizer, label2id, args.max_len)
    val_dataset = NERDataset(val_data, tokenizer, label2id, args.max_len)
    test_dataset = NERDataset(test_data, tokenizer, label2id, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2, shuffle=False)

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    print(f"\n加载模型: {args.model_path}")
    model = BertForTokenClassification(args.model_path, num_labels=len(label2id))
    model = model.to(device)

    # ── 优化器 ────────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # ── 验证 ──────────────────────────────────────────────────────────────
        val_f1, _ = evaluate(model, val_loader, id2label, device)

        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证 F1: {val_f1:.4f}")
        print(f"  耗时: {elapsed:.0f}s")

        log_records.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_f1": val_f1,
            "elapsed_s": elapsed
        })

        # ── 保存最佳模型 ──────────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            tokenizer.save_pretrained(output_dir)
            print(f"  ✓ 最优模型已保存 (F1={val_f1:.4f})")

    # ── 测试 ──────────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("测试集评估")
    print("="*50)
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    test_f1, _ = evaluate(model, test_loader, id2label, device)
    print(f"测试 F1: {test_f1:.4f}")

    # ── 保存日志 ──────────────────────────────────────────────────────────────
    with open(output_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！")
    print(f"模型保存至: {output_dir / 'best_model.pt'}")
    print(f"训练日志: {output_dir / 'train_log.json'}")

if __name__ == "__main__":
    main()
