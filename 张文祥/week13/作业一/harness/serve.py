"""
---
name: harness-serve
description: >-
  FastAPI HTTP 服务，为 Skill Harness 提供 Web 界面后端。
  支持技能列表查询、渐进式执行 SSE 推送、暂停时输入提交。
---

# harness-serve — Skill Harness Web 后端

## 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /skills | 列出所有可用 skill |
| POST | /execute | SSE 渐进式执行 |
| POST | /input | 暂停时提供输入 |
| GET | / | index.html |
"""

import os, sys, json, asyncio, logging, uuid
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from skill_loader import SkillRegistry
from skill_executor import ProgressiveExecutor

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

registry = SkillRegistry()
registry.scan(str(HERE.parent))

sessions: dict[str, ProgressiveExecutor] = {}

app = FastAPI(title="Skill Harness")

class ExecuteRequest(BaseModel):
    skill_name: str
    user_input: str = ""

class InputRequest(BaseModel):
    session_id: str
    user_input: str


@app.get("/skills")
def list_skills():
    return [
        {
            "name": s.name,
            "description": s.description[:200],
            "trigger_patterns": s.trigger_patterns[:5],
            "steps": [{"order": st.order, "action": st.action, "description": st.description} for st in s.steps],
        }
        for s in registry.skills
    ]


@app.post("/execute")
async def execute_skill(req: ExecuteRequest, request: Request):
    skill = registry.find_by_name(req.skill_name)
    if not skill:
        return JSONResponse({"error": f"Skill '{req.skill_name}' not found"}, status_code=404)

    sid = uuid.uuid4().hex[:8]
    executor = ProgressiveExecutor(skill)
    executor.start(context={"user_input": req.user_input, "work_dir": str(HERE.parent)})
    sessions[sid] = executor

    async def event_stream():
        try:
            yield f"data: {json.dumps({'type':'start','session_id':sid,'skill_name':skill.name,'total_steps':len(skill.steps)})}\n\n"
            while True:
                result = executor.step()
                if result is None:
                    break
                payload = {"type": "step", "step": result.step, "action": result.action, "description": result.description, "status": result.status, "output": result.output[:300]}
                if result.status == "awaiting_input":
                    payload["prompt"] = result.prompt
                    yield f"data: {json.dumps(payload)}\n\n"
                    return  # wait for /input
                yield f"data: {json.dumps(payload)}\n\n"
                if result.status in ("completed", "error"):
                    yield f"data: {json.dumps({'type':'done','status':result.status,'output':result.output})}\n\n"
                    sessions.pop(sid, None)
                    return
            yield f"data: {json.dumps({'type':'done','status':'completed'})}\n\n"
        except asyncio.CancelledError:
            executor.cancel()
            sessions.pop(sid, None)
        finally:
            sessions.pop(sid, None)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/input")
async def provide_input(req: InputRequest):
    executor = sessions.get(req.session_id)
    if not executor:
        return JSONResponse({"error": "Session not found or expired"}, status_code=404)
    executor.resume(req.user_input)
    return {"status": "resumed"}


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
