import os, time, json, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_client import tavily_search, format_search_result

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是任务编排主代理。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子代理并行查询（参数=用 | 分隔的多个子任务）

【关键决策原则】
- 只要任务涉及 2 个及以上相互独立的侧面（如「查两个城市天气」「查多个市场数据」等），
  必须用 dispatch_subagents 拆给子代理并行处理，不要自己串行 web_search 多次。
  示例："同时查询北京天气和广州天气" → Action: dispatch_subagents
        Action Input: 北京今天的天气 | 广州今天的天气
  示例："查询2024年新能源汽车销量和特斯拉股价" → Action: dispatch_subagents
        Action Input: 2024年中国新能源汽车销量 | 特斯拉最近股价
- 只有单一事实问题（如"2024年比亚迪销量"）才直接 web_search
- 拿到子代理结果后，综合成结构化报告

【示例】
Question: 查询北京和上海今天的天气
Thought: 这是两个独立子任务，必须派发子代理并行收集，不能自己串行搜索
Action: dispatch_subagents
Action Input: 北京今天的天气 | 上海今天的天气
Observation: 并行查询完成：2 个子代理...
Thought: 已收齐两个维度的并行查询结果，综合成报告
Final Answer: （分任务报告）"""


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    action_input: "子任务1 | 子任务2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（对比凸显并行加速）。"""
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出子任务"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={"web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                                  "联网搜索，参数是查询词")},
            max_steps=4, model_tag="deepseek-chat(子)")
        defs.append((sid, sub, topic))

    dispatch_info = {"subtopics": subtopics,
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results = {}

    def _run_one(sid, sub, topic):
        return sid, sub.run(topic, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    if serial:
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid for sid, sub, topic in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic, "trace": res["trace"],
                    "duration": res["duration"], "final_answer": res["final_answer"]}
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    parts = [f"【子任务: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行查询完成：{len(defs)} 个子代理，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def run_agent(question: str, on_main_step: Callable = None,
              on_subagent_step: Callable = None,
              on_subagent_done: Callable = None,
              on_dispatch: Callable = None,
              serial: bool = False) -> dict:
    """执行一次任务编排。返回 {final_answer, main_trace, subagents, parallel_stats}。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        return _dispatch_subagents(action_input, shared_state=info,
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                           "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子代理并行查询，参数=用 | 分隔的多个子任务"),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    q = "同时查询北京今天的天气和上海今天的天气"
    r = run_agent(q)
    print("\n" + "=" * 60)
    print(f"主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n报告头:\n{r['final_answer'][:200]}")
