import gc
import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── 配置 ──────────────────────────────────────────────────────────────
MODEL_PATH = r"K:\week07\models\Qwen\Qwen2___5-0___5B-Instruct"
OLLAMA_DIR = r"C:\Users\Administrator\AppData\Local\Ollama"
OLLAMA_MODEL = "qwen2.5:0.5b"  # Ollama 中使用的模型名
N_PROMPTS = 50
MAX_NEW_TOKENS = 100
BATCH_SIZE = 8

# ── 测试 prompts（长短混合，模拟真实业务）────────────────────────────
SHORT_QUESTIONS = [
    "什么是股票？", "什么是基金？", "什么是ETF？", "什么是债券？", "什么是期权？",
    "什么是熊市？", "什么是牛市？", "什么是PE？", "什么是ROE？", "什么是毛利率？",
]
MEDIUM_QUESTIONS = [
    "解释一下价值投资和趋势投资的区别。",
    "什么情况下应该止损？",
    "为什么会出现股市崩盘？",
    "沪深300和中证500有什么区别？",
    "什么是量化交易？",
    "基金定投的优势是什么？",
    "股票回购对股价有什么影响？",
    "可转债有哪些特点？",
    "如何判断一家公司是否值得投资？",
    "什么是做市商制度？",
]
LONG_QUESTIONS = [
    "请详细介绍一下巴菲特的投资理念及其核心原则，并举例说明。",
    "解释下现金流折现（DCF）估值法的基本步骤、使用的参数以及它的局限性。",
    "比较A股和美股在交易制度、监管环境、投资者结构等方面的主要差异。",
    "什么是技术分析？它和基本面分析有什么区别？两种方法各自的适用场景是什么？",
    "详细解释资产配置的核心思想，常见的几种配置模型，以及如何根据个人风险偏好调整。",
]
PROMPTS = (SHORT_QUESTIONS * 3 + MEDIUM_QUESTIONS * 1 + LONG_QUESTIONS * 2)[:N_PROMPTS]
assert len(PROMPTS) == N_PROMPTS


# ══════════════════════════════════════════════════════════════════════
#                     模式 A+B: transformers
# ══════════════════════════════════════════════════════════════════════

def bench_transformers(prompts: list[str]) -> dict:
    print("\n" + "=" * 70)
    print("  加载 transformers Qwen2-0.5B-Instruct")
    print("=" * 70)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cpu",
    )
    model.eval()

    # 统一构造 chat prompts
    def make_prompt(q: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )

    chat_prompts = [make_prompt(q) for q in prompts]

    # ── 串行 ────────────────────────────────────────────────────────
    print("\n[A] transformers 串行（一次一条）...")
    total_tokens_a = 0
    t0 = time.time()
    for i, p in enumerate(chat_prompts):
        inputs = tokenizer(p, return_tensors="pt").to("cpu")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        total_tokens_a += len(gen_ids)
        if (i + 1) % 10 == 0:
            print(f"    进度 {i+1}/{len(chat_prompts)}")
    dt_a = time.time() - t0

    # ── batch ──────────────────────────────────────────────────────
    print(f"\n[B] transformers batch={BATCH_SIZE}（手动 padding）...")
    tokenizer.padding_side = "left"  # decoder-only 必须左 padding
    total_tokens_b = 0
    t0 = time.time()
    for i in range(0, len(chat_prompts), BATCH_SIZE):
        batch = chat_prompts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to("cpu")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[:, enc["input_ids"].shape[1]:]
        # 去掉 pad token 计新 token
        for row in gen_ids:
            total_tokens_b += (row != tokenizer.pad_token_id).sum().item()
        print(f"    进度 batch {i//BATCH_SIZE + 1}/{(len(chat_prompts)+BATCH_SIZE-1)//BATCH_SIZE}")
    dt_b = time.time() - t0

    # 释放显存
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "serial":       {"time": dt_a, "gen_tokens": total_tokens_a,
                         "qps": len(prompts) / dt_a,
                         "tps": total_tokens_a / dt_a},
        "batch":        {"time": dt_b, "gen_tokens": total_tokens_b,
                         "qps": len(prompts) / dt_b,
                         "tps": total_tokens_b / dt_b},
    }


# ══════════════════════════════════════════════════════════════════════
#                     模式 C: Ollama (内置 continuous batching)
# ══════════════════════════════════════════════════════════════════════

def bench_ollama(prompts: list[str]) -> dict:
    print("\n" + "=" * 70)
    print("  加载 Ollama (Qwen2-0.5B-Instruct)")
    print("=" * 70)

    import ollama
    from transformers import AutoTokenizer

    os.environ["OLLAMA_MODELS"] = OLLAMA_DIR

    # 确保模型可用（首次会自动拉取）
    try:
        ollama.show(OLLAMA_MODEL)
    except ollama.ResponseError:
        print(f"  正在拉取模型 {OLLAMA_MODEL} ...")
        ollama.pull(OLLAMA_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )
        for q in prompts
    ]

    print(f"\n[C] Ollama 批处理（连续请求，服务端 continuous batching）...")
    total_tokens_c = 0
    t0 = time.time()
    for p in chat_prompts:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=p,
            options={
                "num_predict": MAX_NEW_TOKENS,
                "temperature": 0,
            },
        )
        total_tokens_c += response.get("eval_count", 0)
    dt_c = time.time() - t0

    return {
        "ollama": {"time": dt_c, "gen_tokens": total_tokens_c,
                   "qps": len(prompts) / dt_c,
                   "tps": total_tokens_c / dt_c},
    }


# ══════════════════════════════════════════════════════════════════════
#                     绘图 + 报告
# ══════════════════════════════════════════════════════════════════════

def plot_results(r: dict, out_path: str):
    modes = ["transformers\nserial", f"transformers\nbatch={BATCH_SIZE}", "Ollama\ncontinuous\nbatching"]
    times = [r["serial"]["time"], r["batch"]["time"], r["ollama"]["time"]]
    qps = [r["serial"]["qps"], r["batch"]["qps"], r["ollama"]["qps"]]
    tps = [r["serial"]["tps"], r["batch"]["tps"], r["ollama"]["tps"]]
    colors = ["#aab7c4", "#82b1ff", "#69f0ae"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.rcParams["axes.unicode_minus"] = False

    bars = axes[0].bar(modes, times, color=colors)
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title(f"Total Time for {N_PROMPTS} Requests")
    for b, v in zip(bars, times):
        axes[0].text(b.get_x() + b.get_width()/2, v, f"{v:.1f}s",
                     ha="center", va="bottom")

    bars = axes[1].bar(modes, qps, color=colors)
    axes[1].set_ylabel("QPS (requests/sec)")
    axes[1].set_title("Requests Per Second (higher is better)")
    for b, v in zip(bars, qps):
        axes[1].text(b.get_x() + b.get_width()/2, v, f"{v:.1f}",
                     ha="center", va="bottom")

    bars = axes[2].bar(modes, tps, color=colors)
    axes[2].set_ylabel("Tokens / sec (generated)")
    axes[2].set_title("Generation Throughput (tokens/sec)")
    for b, v in zip(bars, tps):
        axes[2].text(b.get_x() + b.get_width()/2, v, f"{v:.0f}",
                     ha="center", va="bottom")

    plt.suptitle("Ollama vs Transformers: Throughput Benchmark (Qwen2-0.5B, RTX 4060 8GB)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n柱状图已保存：{out_path}")


def main():
    print("=" * 70)
    print(f"  Throughput Benchmark  |  {N_PROMPTS} prompts × max {MAX_NEW_TOKENS} new tokens")
    print("=" * 70)

    tf = bench_transformers(PROMPTS)
    ol = bench_ollama(PROMPTS)
    results = {**tf, **ol}

    # ── 汇总表 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)
    print(f"{'模式':<30}{'总耗时':<12}{'QPS':<10}{'tokens/s':<12}{'相对Ollama':<10}")
    print("-" * 80)
    speedup_base = results["ollama"]["qps"]
    name_map = {"serial": "[A] transformers 串行",
                "batch":  f"[B] transformers batch={BATCH_SIZE}",
                "ollama": "[C] Ollama 批处理"}
    for k in ["serial", "batch", "ollama"]:
        r = results[k]
        rel = r["qps"] / speedup_base
        print(f"{name_map[k]:<28}{r['time']:>6.2f}s     "
              f"{r['qps']:>5.2f}     {r['tps']:>6.0f}      {rel:>5.2f}×")

    # ── 保存结果 ────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "throughput_results.json")
    png_path = os.path.join(out_dir, "throughput_comparison.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_prompts": N_PROMPTS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "batch_size": BATCH_SIZE,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 结果保存：{json_path}")

    plot_results(results, png_path)

    print("\n" + "=" * 70)
    print("  核心结论：")
    print(f"    Ollama 相对 transformers 串行加速：{results['ollama']['qps']/results['serial']['qps']:.1f}×")
    print(f"    Ollama 相对 transformers batch:    {results['ollama']['qps']/results['batch']['qps']:.1f}×")
    print("    关键机制：llama.cpp continuous batching")
    print("=" * 70)


if __name__ == "__main__":
    main()
