import os
import re
import uuid
import shutil
import threading
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

APP_VERSION = "10.0"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "ai_content_factory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 720
HEIGHT = 1280
FPS = 24
MAX_SCRIPT_CHARS = 50000
MAX_VIDEO_SECONDS = 3600  # 1 hour

app = FastAPI(title="AI Content Factory Video Studio", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/downloads", StaticFiles(directory=str(OUTPUT_DIR)), name="downloads")

# NOTE: This in-memory queue is suitable for a simple prototype.
# A Render restart/redeploy clears running jobs and the in-memory job list.
JOBS = {}
JOBS_LOCK = threading.Lock()


class ContentRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=250)
    niche: str = Field(default="General", max_length=150)
    language: str = "hi"
    script: str = Field(min_length=1, max_length=MAX_SCRIPT_CHARS)
    # None means use the full generated narration length.
    duration: int | None = Field(default=None, ge=1, le=MAX_VIDEO_SECONDS)


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        box = draw.textbbox((0, 0), candidate, font=font)
        if box[2] - box[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_scene(topic, narration, scene_number, total_scenes):
    image = Image.new("RGB", (WIDTH, HEIGHT), (12, 15, 38))
    draw = ImageDraw.Draw(image)
    for i in range(15):
        inset = i * 20
        x0 = min(inset, WIDTH // 2 - 2)
        y0 = min(inset, HEIGHT // 2 - 2)
        x1 = max(x0 + 1, WIDTH - inset)
        y1 = max(y0 + 1, HEIGHT - inset)
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=25,
            outline=(min(255, 35 + i * 7), min(255, 40 + i * 4), min(255, 100 + i * 5)),
            width=3,
        )
    draw.ellipse((455, 115, 645, 305), fill=(75, 40, 130))
    draw.ellipse((485, 145, 615, 275), fill=(110, 65, 175))
    draw.ellipse((35, 900, 220, 1085), fill=(35, 75, 135))

    small_font, title_font, subtitle_font = load_font(20), load_font(42), load_font(27)
    draw.text((38, 45), "AI CONTENT FACTORY", font=small_font, fill=(190, 210, 255))
    draw.rounded_rectangle(
        (32, 340, WIDTH - 32, 920),
        radius=32, fill=(23, 28, 65), outline=(120, 100, 210), width=4,
    )
    y = 415
    for line in wrap_text(draw, topic, title_font, WIDTH - 115)[:5]:
        draw.text((58, y), line, font=title_font, fill=(255, 225, 150))
        y += 62
    draw.line((58, 745, WIDTH - 58, 745), fill=(100, 145, 240), width=3)
    y = 785
    for line in wrap_text(draw, narration, subtitle_font, WIDTH - 115)[:3]:
        draw.text((58, y), line, font=subtitle_font, fill=(245, 245, 255))
        y += 42
    draw.text((38, HEIGHT - 70), f"SCENE {scene_number}/{total_scenes}", font=small_font, fill=(190, 210, 255))
    return image


def split_script(script, count):
    # Split at sentence punctuation first; if there are too few sentences,
    # divide the text into approximately equal groups.
    sentences = [s.strip() for s in re.split(r"(?<=[।!?])\s+", script.strip()) if s.strip()]
    if not sentences:
        sentences = [script.strip()]
    groups = [[] for _ in range(max(1, count))]
    for i, sentence in enumerate(sentences):
        index = min(i * len(groups) // len(sentences), len(groups) - 1)
        groups[index].append(sentence)
    result = [" ".join(group).strip() for group in groups]
    return [part for part in result if part] or [script.strip()]


def update_job(job_id, **values):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)
            JOBS[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()


def render_job(job_id, topic, script, language, requested_duration):
    work_dir = OUTPUT_DIR / (job_id + "_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "narration.mp3"
    audio_download_path = OUTPUT_DIR / f"{job_id}.mp3"
    video_path = OUTPUT_DIR / f"{job_id}.mp4"
    audio_clip = None
    scene_clips = []
    final_clip = None

    try:
        update_job(job_id, status="processing", progress=5, message="Creating narration audio")
        gTTS(text=script, lang=language, slow=False).save(str(audio_path))
        audio_clip = AudioFileClip(str(audio_path))
        audio_duration = float(audio_clip.duration or 0)

        if audio_duration <= 0:
            raise RuntimeError("Narration audio duration could not be detected.")
        if audio_duration > MAX_VIDEO_SECONDS:
            raise RuntimeError("Generated narration exceeds the 1-hour limit.")

        # Auto mode follows the entire narration. If a duration is explicitly
        # selected, never truncate the narration; use whichever is longer.
        target_duration = max(audio_duration, float(requested_duration or 0))

        # Use one scene about every 10 seconds, capped at 360 scenes/hour.
        scene_count = max(1, min(360, int((target_duration + 9) // 10)))
        scene_texts = split_script(script, scene_count)
        scene_duration = target_duration / len(scene_texts)

        update_job(job_id, progress=15, message="Preparing video scenes",
                   duration_seconds=round(target_duration, 1))
        for index, narration_part in enumerate(scene_texts, start=1):
            image = make_scene(topic, narration_part, index, len(scene_texts))
            image_path = work_dir / f"scene_{index}.jpg"
            image.save(image_path, quality=90)
            scene_clips.append(ImageClip(str(image_path)).with_duration(scene_duration))
            if index % 10 == 0 or index == len(scene_texts):
                progress = 15 + int(45 * index / len(scene_texts))
                update_job(job_id, progress=progress,
                           message=f"Preparing scene {index} of {len(scene_texts)}")

        update_job(job_id, progress=65, message="Rendering MP4 video")
        final_clip = concatenate_videoclips(scene_clips, method="compose")
        final_clip = final_clip.with_audio(audio_clip)
        final_clip.write_videofile(
            str(video_path), fps=FPS, codec="libx264", audio_codec="aac",
            preset="ultrafast", threads=2, logger=None,
        )

        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RuntimeError("The MP4 file was not created.")
        shutil.copy2(audio_path, audio_download_path)

        base_url = os.getenv("PUBLIC_BASE_URL", "https://ai-content-backend-kgew.onrender.com").rstrip("/")
        update_job(
            job_id,
            status="completed",
            progress=100,
            message="MP4 rendered successfully",
            topic=topic,
            duration_seconds=round(target_duration, 1),
            video_url=f"{base_url}/downloads/{job_id}.mp4",
            audio_url=f"{base_url}/downloads/{job_id}.mp3",
            storyboard=[
                {"scene": i + 1, "visual": "3D-look layered scene", "narration": part}
                for i, part in enumerate(scene_texts)
            ],
        )
    except Exception as exc:
        update_job(job_id, status="failed", progress=100, message=str(exc), error=str(exc))
    finally:
        if final_clip is not None:
            try:
                final_clip.close()
            except Exception:
                pass
        if audio_clip is not None:
            try:
                audio_clip.close()
            except Exception:
                pass
        for clip in scene_clips:
            try:
                clip.close()
            except Exception:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/")
def health_check():
    return {"status": "Online", "engine": "AI Content Factory Video Studio", "version": APP_VERSION}


@app.post("/generate-video")
def generate_content(req: ContentRequest):
    topic = req.topic.strip()
    script = req.script.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required.")
    if not script:
        raise HTTPException(status_code=400, detail="Script is required.")
    if req.language != "hi":
        raise HTTPException(status_code=400, detail="This version supports Hindi narration only.")

    job_id = "JOB-" + uuid.uuid4().hex[:10].upper()
    with JOBS_LOCK:
        JOBS[job_id] = {
            "success": True,
            "job_id": job_id,
            "topic": topic,
            "status": "queued",
            "progress": 0,
            "message": "Video job queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    thread = threading.Thread(
        target=render_job,
        args=(job_id, topic, script, req.language, req.duration),
        daemon=True,
    )
    thread.start()
    return {"success": True, "job_id": job_id, "status": "queued", "message": "Video rendering started."}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(
                status_code=404,
                detail="Job not found. The service may have restarted; please submit the video again."
            )
        return dict(job)
