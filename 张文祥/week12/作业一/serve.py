import os
import sys
import json
import logging
import asyncio
from uuid import uuid4
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 对话会话存储，key=session_id, value=messages 列表
sessions: dict[str, list] = {}

# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:   str
    max_steps:  int = 10
    session_id: str = ""  # 客户端传入 session_id，空串表示新建会话


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(req: QueryRequest, mode: str):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。

    支持 session_id 实现多轮对话：
      1. 从 sessions 中取出历史消息
      2. 传入 react_run 的 history 参数
      3. 流结束后将更新后的消息写回 sessions
    """
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    # session 管理：新建或续接已有对话
    session_id = req.session_id or str(uuid4())
    messages = sessions.get(session_id)  # 取历史，None 则新建
    if messages is None:
        messages = []

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            # 传入 history=messages，messages 会被原地修改
            for step_data in react_run(req.question, max_steps=req.max_steps, history=messages):
                queue.put_nowait(step_data)
        finally:
            queue.put_nowait(_SENTINEL)

    # 返回 session_id 给前端，后续请求携带此 id
    yield _sse({"type": "start", "question": req.question, "mode": mode, "session_id": session_id})

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        yield _sse(step_data)

    # 保存更新后的消息到 sessions
    sessions[session_id] = messages
    yield _sse({"type": "done", "session_id": session_id})


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req, "manual"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req, "fc"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
