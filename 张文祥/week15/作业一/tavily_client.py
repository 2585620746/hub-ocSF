import os, logging
from tavily import TavilyClient

logger = logging.getLogger(__name__)
_client = None


TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")


def get_client() -> TavilyClient:
    """单例 TavilyClient。"""
    global _client
    if _client is None:
        _client = TavilyClient(api_key=TAVILY_KEY)
    return _client


def tavily_search(query: str, max_results: int = 5) -> dict:
    """调用 Tavily 搜索。返回 {answer, results, response_time}。
    失败返回 {"error": ...}，不抛异常（ReAct loop 兜底）。"""
    try:
        client = get_client()
        data = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=True,
        )
        results = [{"title": r.get("title", ""), "url": r.get("url", ""),
                     "content": (r.get("content") or "")[:600]}
                    for r in data.get("results", [])]
        return {"answer": data.get("answer") or "",
                "results": results,
                "response_time": data.get("response_time")}
    except Exception as e:
        logger.warning(f"Tavily 搜索失败 '{query}': {e}")
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}


def format_search_result(r: dict) -> str:
    """把 Tavily 返回格式化成喂给 LLM 的文本。"""
    if "error" in r:
        return f"搜索失败: {r['error']}"
    parts = []
    if r.get("answer"):
        parts.append(f"摘要: {r['answer']}")
    for i, res in enumerate(r.get("results", []), 1):
        parts.append(f"[{i}] {res['title']}\n    {res['content'][:300]}")
    return "\n".join(parts) if parts else "无结果"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = tavily_search("中国新能源汽车2024年销量")
    print(format_search_result(r)[:400])
