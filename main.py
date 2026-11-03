import os, uuid, sqlite3, subprocess, re
from pathlib import Path
from datetime import datetime, timezone
from threading import Thread
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from gtts import gTTS
import imageio_ffmpeg
from scene_engine import router as scene_router
from image_engine import router as image_router

app = FastAPI(title="Personal AI Video Studio", version="1.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "video_jobs.db"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.include_router(scene_router)
app.include_router(image_router)

class AudioRequest(BaseModel):
    script: str = Field(min_length=1, max_length=10000)
    language: str = "hi"
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0)

class VideoJobRequest(BaseModel):
    script: str = Field(min_length=1, max_length=50000)
    language: str = "hi"
    video_format: str = "shorts"
    resolution: str = "720p"

def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

@app.on_event("startup")
def startup():
    with db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS video_jobs (
            job_id TEXT PRIMARY KEY, status TEXT NOT NULL, progress INTEGER NOT NULL,
            script TEXT NOT NULL, language TEXT NOT NULL, video_format TEXT NOT NULL,
            resolution TEXT NOT NULL, video_url TEXT, error TEXT, created_at TEXT NOT NULL)''')

def update_job(job_id, status, progress, video_url=None, error=None):
    with db() as conn:
        conn.execute("UPDATE video_jobs SET status=?, progress=?, video_url=?, error=? WHERE job_id=?",
                     (status, progress, video_url, error, job_id))

def render_video(job_id, script, language, video_format, resolution):
    audio_path, video_path = OUTPUT_DIR / f"{job_id}.mp3", OUTPUT_DIR / f"{job_id}.mp4"
    try:
        update_job(job_id, "processing", 10)
        gTTS(text=script, lang=language, slow=False).save(str(audio_path))
        update_job(job_id, "processing", 40)
        width, height = (720, 1280) if video_format == "shorts" else (1280, 720)
        if resolution == "1080p": width, height = int(width*1.5), int(height*1.5)
        result = subprocess.run([FFMPEG, "-i", str(audio_path), "-hide_banner"], capture_output=True, text=True)
        m = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", result.stderr)
        if not m: raise RuntimeError("Could not read audio duration")
        duration = int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3)) + int(m.group(4))/100
        update_job(job_id, "processing", 60)
        cmd = [FFMPEG, "-y", "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration}",
               "-i", str(audio_path), "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
               "-tune", "stillimage", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(video_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        update_job(job_id, "completed", 100, f"/outputs/{job_id}.mp4")
    except Exception as exc:
        update_job(job_id, "failed", 0, error=str(exc))
    finally:
        if audio_path.exists(): audio_path.unlink()

@app.get("/")
def home(): return {"success": True, "engine": "Personal AI Video Studio", "status": "online", "version": "1.4.0"}

@app.get("/api/health")
def health(): return {"status": "healthy", "version": "1.4.0"}

@app.post("/api/generate-audio")
def generate_audio(data: AudioRequest):
    if data.language not in ["hi", "mr", "en"]: raise HTTPException(400, "Supported languages: hi, mr, en")
    try:
        name = f"{uuid.uuid4().hex}.mp3"; gTTS(text=data.script, lang=data.language, slow=False).save(str(OUTPUT_DIR/name))
        return {"success": True, "language": data.language, "audio_url": f"/outputs/{name}", "download_url": f"/outputs/{name}"}
    except Exception as exc: raise HTTPException(500, str(exc))

@app.post("/api/video/jobs")
def create_video_job(data: VideoJobRequest):
    if data.language not in ["hi", "mr", "en"]: raise HTTPException(400, "Supported languages: hi, mr, en")
    if data.video_format not in ["shorts", "landscape"]: raise HTTPException(400, "Format must be shorts or landscape")
    if data.resolution not in ["720p", "1080p"]: raise HTTPException(400, "Resolution must be 720p or 1080p")
    jid, created = str(uuid.uuid4()), datetime.now(timezone.utc).isoformat()
    with db() as conn:
        conn.execute("INSERT INTO video_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (jid, "queued", 0, data.script, data.language, data.video_format, data.resolution, None, None, created))
    Thread(target=render_video, args=(jid, data.script, data.language, data.video_format, data.resolution), daemon=True).start()
    return {"success": True, "job_id": jid, "status": "queued"}

@app.get("/api/video/jobs/{job_id}")
def get_video_job(job_id: str):
    with db() as conn: row = conn.execute("SELECT * FROM video_jobs WHERE job_id=?", (job_id,)).fetchone()
    if not row: raise HTTPException(404, "Video job not found")
    return {"success": True, "job": dict(row)}

@app.get("/api/video/jobs")
def get_all_video_jobs():
    with db() as conn: rows = conn.execute("SELECT * FROM video_jobs ORDER BY created_at DESC").fetchall()
    return {"success": True, "total_jobs": len(rows), "jobs": [dict(x) for x in rows]}
