import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI(title="AI Content Factory Autonomous Cloud Worker", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    topic: str
    niche: str = "Emotional Stories"
    style: str = "3D Animation"

@app.get("/")
def health_check():
    return {"status": "Online", "engine": "Autonomous Cloud FFmpeg Worker v3.0"}

@app.post("/generate-video")
async def generate_video(req: VideoRequest):
    try:
        # Step 1: Generate Clean Script & Prompts
        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required.")
        
        # Step 2: Fetch 3D Character Visual URLs (Pollinations Open-Source AI Endpoint)
        prompts = [
            f"{topic} 3d pixar style cute character emotional expression cinematic lighting",
            f"{topic} dramatic scene 3d animation digital art masterpiece",
            f"{topic} touching moment emotional storytelling 3d render"
        ]
        
        image_urls = []
        for idx, p in enumerate(prompts):
            img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(p)}?width=1080&height=1920&nologo=true&seed={600+idx}"
            image_urls.append(img_url)

        # Step 3: Return complete production package manifest to client/frontend
        return {
            "success": True,
            "job_id": f"CLOUD-JOB-{os.urandom(4.hex())}",
            "topic": topic,
            "style": req.style,
            "visual_assets": image_urls,
            "status": "Ready for client rendering & audio sync"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))