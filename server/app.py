from fastapi import FastAPI

app = FastAPI(title="FightingOverlay Server", version="0.1.0")


@app.post("/athletes")
async def create_athlete() -> dict[str, str]:
    return {"status": "stub", "message": "create athlete"}


@app.post("/clips")
async def create_clip() -> dict[str, str]:
    return {"status": "stub", "message": "create clip"}


@app.post("/clips/{clip_id}/upload")
async def upload_clip(clip_id: str) -> dict[str, str]:
    return {"status": "stub", "clip_id": clip_id}


@app.post("/clips/{clip_id}/process")
async def process_clip(clip_id: str) -> dict[str, str]:
    return {"status": "stub", "clip_id": clip_id}


@app.get("/clips/{clip_id}/status")
async def clip_status(clip_id: str) -> dict[str, str]:
    return {"status": "stub", "clip_id": clip_id}
