
import os
import uuid
import sqlite3
import subprocess
import re
import asyncio

from pathlib import Path
from datetime import datetime, timezone
from threading import Thread

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from gtts import gTTS
import edge_tts
import imageio_ffmpeg

from scene_engine import (
    router as scene_router,
    SceneRequest,
    SCENE_JOBS,
    create_scene_plan,
)
from image_engine import (
    router as image_router,
    ImageRequest,
    generate_scene_images,
    worker_health,
)
from video_engine import (
    video_worker_health,
    generate_scene_videos,
    compose_cinematic_video,
)


# ==========================================================
# 1. APPLICATION CONFIGURATION
# ==========================================================

app = FastAPI(
    title="Personal AI Video Studio - Cinematic AI Video",
    version="4.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = BASE_DIR / "video_jobs.db"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs",
)

app.include_router(scene_router)
app.include_router(image_router)


# ==========================================================
# 2. REQUEST MODELS
# ==========================================================

class VideoJobRequest(BaseModel):
    script: str = Field(min_length=1, max_length=50000)
    language: str = "hi"
    video_format: str = "shorts"
    resolution: str = "720p"

    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    voice_gender: str = "female"

    character_description: str = ""

    visual_style: str = (
        "photorealistic live-action Indian cinema, "
        "cinematic natural lighting"
    )

    image_width: int = Field(default=576, ge=256, le=1024)
    image_height: int = Field(default=1024, ge=256, le=1536)

    image_strength: float = Field(default=0.62, ge=0.2, le=0.9)

    generate_character_reference: bool = True
    render_mode: str = "cinematic"

    target_duration: int = Field(default=60, ge=5, le=3600)


class LegacyVideoRequest(BaseModel):
    script: str = Field(min_length=1, max_length=50000)

    language: str = "hi"
    aspect_ratio: str = "9:16"
    duration: int | None = None

    niche: str = "Hindi emotional human stories"

    character_description: str = ""

    visual_style: str = (
        "photorealistic live-action Indian cinema, "
        "cinematic natural lighting"
    )

    voice_gender: str = "female"
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0)

    render_mode: str = "cinematic"


# ==========================================================
# 3. DATABASE
# ==========================================================

def db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


@app.on_event("startup")
def startup():
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS video_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                script TEXT NOT NULL,
                language TEXT NOT NULL,
                video_format TEXT NOT NULL,
                resolution TEXT NOT NULL,
                video_url TEXT,
                error TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        cols = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(video_jobs)"
            ).fetchall()
        }

        additional_columns = [
            ("duration_seconds", "REAL"),
            ("scenes_count", "INTEGER"),
            ("image_provider", "TEXT"),
        ]

        for name, column_type in additional_columns:
            if name not in cols:
                conn.execute(
                    f"ALTER TABLE video_jobs ADD COLUMN {name} {column_type}"
                )


def update_job(
    job_id,
    status,
    progress,
    video_url=None,
    error=None,
    duration=None,
    scenes_count=None,
    image_provider=None,
):
    with db() as conn:
        conn.execute(
            """
            UPDATE video_jobs
            SET
                status=?,
                progress=?,
                video_url=COALESCE(?, video_url),
                error=?,
                duration_seconds=COALESCE(?, duration_seconds),
                scenes_count=COALESCE(?, scenes_count),
                image_provider=COALESCE(?, image_provider)
            WHERE job_id=?
            """,
            (
                status,
                progress,
                video_url,
                error,
                duration,
                scenes_count,
                image_provider,
                job_id,
            ),
        )


# ==========================================================
# 4. AUDIO GENERATION
# ==========================================================

def audio_duration(path):
    result = subprocess.run(
        [FFMPEG, "-i", str(path), "-hide_banner"],
        capture_output=True,
        text=True,
    )

    match = re.search(
        r"Duration: (\d+):(\d+):(\d+)\.(\d+)",
        result.stderr,
    )

    if not match:
        raise RuntimeError(
            "Could not read generated voice-over duration."
        )

    return (
        int(match.group(1)) * 3600
        + int(match.group(2)) * 60
        + int(match.group(3))
        + int(match.group(4)) / 100
    )


def make_speech(
    text,
    language,
    voice_gender,
    voice_speed,
    output_path,
):
    gender = (voice_gender or "female").lower()

    if gender not in ("female", "male"):
        raise ValueError("voice_gender must be female or male")

    voice = {
        "hi": {
            "female": "hi-IN-SwaraNeural",
            "male": "hi-IN-MadhurNeural",
        },
        "en": {
            "female": "en-IN-NeerjaNeural",
            "male": "en-IN-PrabhatNeural",
        },
    }.get(language, {}).get(gender)

    if voice:
        rate_pct = max(
            -50,
            min(50, round((voice_speed - 1) * 100)),
        )

        try:
            asyncio.run(
                edge_tts.Communicate(
                    text,
                    voice,
                    rate=f"{rate_pct:+d}%",
                ).save(str(output_path))
            )

            if (
                output_path.exists()
                and output_path.stat().st_size > 1000
            ):
                return {
                    "engine": "edge-tts",
                    "voice": voice,
                }

        except Exception:
            if output_path.exists():
                output_path.unlink()

    gTTS(
        text=text,
        lang=language,
        slow=False,
    ).save(str(output_path))

    return {
        "engine": "gTTS",
        "voice": "default",
    }


# ==========================================================
# 5. SLIDESHOW VIDEO COMPOSITION
# ==========================================================

def compose_slideshow(
    job_id,
    scene_job,
    audio_path,
    video_format,
    resolution,
    duration,
):
    scenes = scene_job["scenes"]

    if not scenes or any(
        scene.get("image_status") != "completed"
        or not scene.get("image_url")
        for scene in scenes
    ):
        raise RuntimeError(
            "Realistic scene images are incomplete. "
            "Video rendering was stopped to prevent a black-screen video."
        )

    width, height = (
        (720, 1280)
        if video_format == "shorts"
        else (1280, 720)
    )

    if resolution == "1080p":
        width = int(width * 1.5)
        height = int(height * 1.5)

    weights = [
        max(1, len(scene.get("scene_text", "").split()))
        for scene in scenes
    ]

    total = sum(weights)
    durations = [
        duration * weight / total
        for weight in weights
    ]

    cmd = [FFMPEG, "-y"]

    for scene, scene_duration in zip(scenes, durations):
        image_path = OUTPUT_DIR / Path(
            scene["image_url"]
        ).name

        if not image_path.exists():
            raise RuntimeError(
                f"Scene image file is missing: {image_path.name}"
            )

        cmd += [
            "-loop", "1",
            "-framerate", "25",
            "-t", f"{scene_duration:.3f}",
            "-i", str(image_path),
        ]

    audio_index = len(scenes)
    cmd += ["-i", str(audio_path)]

    filters = []
    concat_inputs = []

    for i in range(len(scenes)):
        filters.append(
            f"[{i}:v]"
            f"scale={width}:{height}:"
            "force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            "zoompan=z='min(zoom+0.00035,1.07)':"
            f"d=1:s={width}x{height}:fps=25,"
            "setsar=1,format=yuv420p"
            f"[v{i}]"
        )

        concat_inputs.append(f"[v{i}]")

    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(scenes)}:v=1:a=0[vout]"
    )

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map", "[vout]",
        "-map", f"{audio_index}:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        str(OUTPUT_DIR / f"{job_id}.mp4"),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg video composition failed: "
            + (
                result.stderr[-1800:]
                if result.stderr
                else "unknown error"
            )
        )


# ==========================================================
# 6. VIDEO RENDERING PIPELINE
# ==========================================================

def render_video(job_id, data):
    audio_path = OUTPUT_DIR / f"{job_id}.mp3"

    try:
        update_job(job_id, "processing", 5)

        image_health = worker_health()

        if not image_health.get("success"):
            raise RuntimeError(
                "GPU worker is not reachable: "
                + image_health.get("error", "unknown worker error")
            )

        if data.render_mode == "cinematic":
            video_health = video_worker_health()

            if not video_health.get("success"):
                raise RuntimeError(
                    "Cinematic video generation is unavailable: "
                    + video_health.get("error", "unknown video worker error")
                )

        update_job(job_id, "processing", 10)

        plan = create_scene_plan(
            SceneRequest(
                script=data.script,
                language=data.language,
                video_format=data.video_format,
                character_description=data.character_description,
                visual_style=data.visual_style,
                target_duration=data.target_duration,
            )
        )

        scene_job = plan["job"]

        update_job(
            job_id,
            "processing",
            18,
            scenes_count=len(scene_job["scenes"]),
        )

        make_speech(
            data.script,
            data.language,
            data.voice_gender,
            data.voice_speed,
            audio_path,
        )

        duration = audio_duration(audio_path)

        update_job(
            job_id,
            "processing",
            28,
            duration=duration,
        )

        if data.render_mode == "cinematic":

            clips = generate_scene_videos(
                job_id,
                scene_job,
                duration,
            )

            update_job(
                job_id,
                "processing",
                82,
                scenes_count=len(clips),
                image_provider="ltx_video_gpu_worker",
            )

            compose_cinematic_video(
                job_id,
                clips,
                audio_path,
                data.video_format,
                data.resolution,
                duration,
                FFMPEG,
            )

        elif data.render_mode == "slideshow":

            image_result = generate_scene_images(
                scene_job["job_id"],
                ImageRequest(
                    width=data.image_width,
                    height=data.image_height,
                    strength=data.image_strength,
                    generate_character_reference=data.generate_character_reference,
                ),
            )

            update_job(
                job_id,
                "processing",
                68,
                scenes_count=image_result["images_generated"],
                image_provider="existing_gpu_worker",
            )

            compose_slideshow(
                job_id,
                scene_job,
                audio_path,
                data.video_format,
                data.resolution,
                duration,
            )

        else:
            raise RuntimeError(
                "render_mode must be cinematic or slideshow"
            )

        update_job(
            job_id,
            "completed",
            100,
            video_url=f"/outputs/{job_id}.mp4",
            duration=duration,
            scenes_count=len(scene_job["scenes"]),
            image_provider=(
                "ltx_video_gpu_worker"
                if data.render_mode == "cinematic"
                else "existing_gpu_worker"
            ),
        )

    except Exception as exc:
        update_job(
            job_id,
            "failed",
            0,
            error=str(exc),
        )

    finally:
        if audio_path.exists():
            audio_path.unlink()


# ==========================================================
# 7. JOB QUEUE
# ==========================================================

def queue_job(data):

    if data.language not in ["hi", "mr", "en"]:
        raise HTTPException(
            400,
            "Supported languages: hi, mr, en",
        )

    if data.voice_gender.lower() not in ["female", "male"]:
        raise HTTPException(
            400,
            "voice_gender must be female or male",
        )

    if data.video_format not in ["shorts", "landscape"]:
        raise HTTPException(
            400,
            "video_format must be shorts or landscape",
        )

    if data.resolution not in ["720p", "1080p"]:
        raise HTTPException(
            400,
            "resolution must be 720p or 1080p",
        )

    if data.render_mode not in ["cinematic", "slideshow"]:
        raise HTTPException(
            400,
            "render_mode must be cinematic or slideshow",
        )

    health = worker_health()

    if not health.get("success"):
        raise HTTPException(
            503,
            "GPU worker is not ready: "
            + health.get("error", "not configured"),
        )

    if data.render_mode == "cinematic":
        video_health = video_worker_health()

        if not video_health.get("success"):
            raise HTTPException(
                503,
                "Cinematic video generation is not ready: "
                + video_health.get("error", "unknown error"),
            )

    job_id = str(uuid.uuid4())
    created = datetime.now(timezone.utc).isoformat()

    with db() as conn:
        conn.execute(
            """
            INSERT INTO video_jobs (
                job_id,
                status,
                progress,
                script,
                language,
                video_format,
                resolution,
                video_url,
                error,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                "queued",
                0,
                data.script,
                data.language,
                data.video_format,
                data.resolution,
                None,
                None,
                created,
            ),
        )

    Thread(
        target=render_video,
        args=(job_id, data),
        daemon=True,
    ).start()

    return {
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "message": (
            "Cinematic moving AI scene generation "
            "and video rendering started."
        ),
    }


# ==========================================================
# 8. RENDER HEALTH CHECK
# ==========================================================

@app.get("/")
def home():
    return {
        "success": True,
        "engine": "Personal AI Video Studio",
        "status": "online",
        "version": "4.0.1",
        "video_pipeline": (
            "LTX moving AI scene clips + Hindi voice + cinematic MP4"
        ),
    }


@app.head("/")
def home_head():
    return None


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "version": "4.0.1",
        "gpu_worker": worker_health(),
        "video_worker": video_worker_health(),
    }


@app.get("/api/worker/health")
def gpu_health():
    return worker_health()


@app.get("/api/video/worker/health")
def video_gpu_health():
    return video_worker_health()


# ==========================================================
# 9. VIDEO API ENDPOINTS
# ==========================================================

@app.post("/api/video/jobs")
def create_video_job(data: VideoJobRequest):
    return queue_job(data)


@app.get("/api/video/jobs/{job_id}")
def get_video_job(job_id: str):

    with db() as conn:
        row = conn.execute(
            "SELECT * FROM video_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()

    if not row:
        raise HTTPException(
            404,
            "Video job not found",
        )

    job = dict(row)

    return {
        "success": True,
        "job": job,
        "status": job["status"],
        "progress": job["progress"],
        "video_url": job["video_url"],
        "duration_seconds": job["duration_seconds"],
        "error": job["error"],
    }


@app.get("/api/video/jobs")
def get_all_video_jobs():

    with db() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM video_jobs
            ORDER BY created_at DESC
            """
        ).fetchall()

    return {
        "success": True,
        "total_jobs": len(rows),
        "jobs": [dict(row) for row in rows],
    }


# ==========================================================
# 10. LEGACY APPS SCRIPT COMPATIBILITY
# ==========================================================

@app.post("/generate-video")
def legacy_generate_video(data: LegacyVideoRequest):

    video_format = (
        "shorts"
        if data.aspect_ratio in ["9:16", "9:16 vertical"]
        else "landscape"
    )

    request_data = VideoJobRequest(
        script=data.script,
        language=data.language,
        video_format=video_format,
        resolution="720p",
        voice_gender=data.voice_gender,
        voice_speed=data.voice_speed,
        character_description=data.character_description,
        visual_style=data.visual_style,
        render_mode=data.render_mode,
        target_duration=data.duration or 60,
    )

    return queue_job(request_data)


@app.get("/jobs/{job_id}")
def legacy_get_job(job_id: str, request: Request):

    result = get_video_job(job_id)

    video_url = result["video_url"]

    if video_url and video_url.startswith("/"):
        public_base = os.getenv(
            "PUBLIC_BACKEND_URL",
            str(request.base_url).rstrip("/"),
        )

        video_url = (
            public_base.rstrip("/") + video_url
        )

    return {
        "success": True,
        "job_id": job_id,
        "status": result["status"],
        "progress": result["progress"],
        "video_url": video_url,
        "duration_seconds": result["duration_seconds"],
        "error": result["error"],
        "job": result["job"],
    }
