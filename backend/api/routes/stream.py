import asyncio
import json
import os
import sys

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarking.run_benchmarks import load_data
from optimization.optimizer import run_fleet_optimization

router = APIRouter(prefix="/api", tags=["Realtime Demo"])

@router.get("/convergence-stream")
async def convergence_stream(fleet_size: int = 6, generations: int = 10):
    """SSE stream for a live judge-facing convergence view."""
    vessels, routes = load_data()
    result = run_fleet_optimization(vessels[:fleet_size], routes, algorithm="ALL", pop_size=10, generations=generations)

    async def events():
        histories = {name: value.get("convergence_history", []) for name, value in result["all_results"].items()}
        for generation in range(generations):
            payload = {"generation": generation + 1, "algorithms": {name: history[generation] for name, history in histories.items() if generation < len(history)}}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.15)
        yield f"data: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})