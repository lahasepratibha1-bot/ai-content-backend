
import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from gtts import gTTS

app = FastAPI(
    title="Personal AI Video Studio",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs"
)

# Temporary job storage (MVP)
VIDEO_JOBS = {}
JOBS_LOCK = Lock()


class AudioRequest(BaseModel):
    script: str = Field(min_length=1, max_length=10000)
    language: str = "hi"
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0)


class VideoJobRequest(BaseModel):
    script: str = Field(min_length=1, max_length=50000)
    language: str = "hi"
    video_format: str = "shorts"
    resolution: str = "720p"


@app.get("/")
def home():
    return {
        "success": True,
        "engine": "Personal AI Video Studio",
        "status": "online",
        "version": "1.1.0"
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}


# AUDIO GENERATION
@app.post("/api/generate-audio")
def generate_audio(data: AudioRequest):

    if data.language not in ["hi", "mr", "en"]:
        raise HTTPException(
            status_code=400,
            detail="Supported languages: hi, mr, en"
        )

    try:
        filename = f"{uuid.uuid4().hex}.mp3"
        audio_path = OUTPUT_DIR / filename

        tts = gTTS(
            text=data.script,
            lang=data.language,
            slow=False
        )

        tts.save(str(audio_path))

        return {
            "success": True,
            "language": data.language,
            "audio_url": f"/outputs/{filename}",
            "download_url": f"/outputs/{filename}",
            "message": "Audio generated successfully."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# CREATE VIDEO JOB
@app.post("/api/video/jobs")
def create_video_job(data: VideoJobRequest):

    if data.language not in ["hi", "mr", "en"]:
        raise HTTPException(
            status_code=400,
            detail="Supported languages: hi, mr, en"
        )

    if data.video_format not in ["shorts", "landscape"]:
        raise HTTPException(
            status_code=400,
            detail="Format must be shorts or landscape"
        )

    if data.resolution not in ["720p", "1080p"]:
        raise HTTPException(
            status_code=400,
            detail="Resolution must be 720p or 1080p"
        )

    job_id = str(uuid.uuid4())

    job = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "script": data.script,
        "language": data.language,
        "video_format": data.video_format,
        "resolution": data.resolution,
        "video_url": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    with JOBS_LOCK:
        VIDEO_JOBS[job_id] = job

    return {
        "success": True,
        "message": "Video job created successfully.",
        "job": job
    }


# GET ONE VIDEO JOB
@app.get("/api/video/jobs/{job_id}")
def get_video_job(job_id: str):

    with JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Video job not found."
        )

    return {
        "success": True,
        "job": job
    }


# GET ALL VIDEO JOBS
@app.get("/api/video/jobs")
def get_all_video_jobs():

    with JOBS_LOCK:
        jobs = list(VIDEO_JOBS.values())

    jobs.sort(
        key=lambda x: x["created_at"],
        reverse=True
    )

    return {
        "success": True,
        "total_jobs": len(jobs),
        "jobs": jobs
    }
