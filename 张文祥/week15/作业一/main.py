import sys, os, logging, time

logging.basicConfig(level=logging.WARNING)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from agent import run_agent


def red(s): return f"\033[31m{s}\033[0m"
def cyan(s): return f"\033[36m{s}\033[0m"
def green(s): return f"\033[32m{s}\033[0m"
def dim(s): return f"\033[2m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"


def main():
    argv = sys.argv[1:]
    if argv:
        question = " ".join(argv)
        run_one(question)
        return

    print(cyan("══ 并行 Subagent Agent ══"))
    print(dim("示例: 同时查询北京和上海的天气 / 查询汽车销量和特斯拉股价 / 输入 q 退出\n"))
    while True:
        try:
            q = input(yellow("请输入任务: ")).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not q:
            continue
        if q.lower() in ("q", "quit", "exit"):
            break
        run_one(q)


def run_one(question):
    print(f"\n{cyan('─' * 60)}")
    print(f"{cyan('任务: ')}{question}")
    print(f"{cyan('─' * 60)}\n")

    t0 = time.time()

    def on_main_step(step):
        if step.get("action"):
            print(f"{green('[主]')} Thought: {step['thought'][:80]}")
            print(f"      Action: {step['action']} → {step.get('action_input','')[:80]}")

    def on_subagent_step(sid, step):
        if step.get("action") and step["action"] != "Final Answer":
            print(f"{yellow(f'[{sid}]')} Thought: {step['thought'][:60]}")
            print(f"      Action: {step['action']} → {step.get('action_input','')[:60]}")

    def on_subagent_done(sid, duration, topic):
        print(f"  {dim(f'✓ {sid} 完成「{topic}」(用时{duration}s)')}")

    def on_dispatch(info):
        subs = ", ".join(info["subtopics"])
        print(f"  {cyan('⇢ 派发')} {len(info['subtopics'])} 个子代理并行: {subs}")

    r = run_agent(question,
                  on_main_step=on_main_step,
                  on_subagent_step=on_subagent_step,
                  on_subagent_done=on_subagent_done,
                  on_dispatch=on_dispatch)

    total = round(time.time() - t0, 2)
    print(f"\n{green('═ 最终报告 ═')} (总用时 {total}s)")
    print(f"{r['final_answer']}")

    if r["parallel_stats"]:
        st = r["parallel_stats"][-1]
        n = st["n_subagents"]
        wall = st["wall_clock"]
        ser = st["serial_sum"]
        spd = st["speedup"]
        print(f"\n{dim(f'并行统计: {n} 子代理 | 并行墙钟 {wall}s | 串行基线 {ser}s | 加速 {spd}x')}")


if __name__ == "__main__":
    main()
