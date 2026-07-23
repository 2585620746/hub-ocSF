import os
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",  default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--interactive", action="store_true", help="交互式多轮对话模式")
    args = parser.parse_args()

    if args.mode == "manual":
        from react_manual import run_and_print
    else:
        from react_function_calling import run_and_print

    if args.interactive:
        print("=== ReAct Agent 多轮对话模式 ===")
        print("输入 'exit' 退出，'clear' 重置上下文")
        history = None
        while True:
            question = input("\n> ").strip()
            if not question:
                continue
            if question.lower() == "exit":
                break
            if question.lower() == "clear":
                history = None
                print("对话历史已清空")
                continue
            # 传入 history 并接收更新后的 history
            history = run_and_print(question, args.max_steps, history=history)
    else:
        run_and_print(args.question, args.max_steps)
