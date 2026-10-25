import os
import re
import uuid
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

APP_VERSION = "9.1"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "ai_content_factory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 720
HEIGHT = 1280
FPS = 24

app = FastAPI(
    title="AI Content Factory 3D Video Studio",
    version=APP_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount(
    "/downloads",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="downloads"
)


class ContentRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=250)
    niche: str = Field(default="General", max_length=150)
    language: str = "hi"
    script: str = Field(default="", max_length=4500)
    duration: int = Field(default=60, ge=10, le=60)


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
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
    lines = []
    current = ""

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

    # Layered 3D-look frames. Coordinates stay inside the canvas.
    for i in range(15):
        inset = i * 20
        x0 = min(inset, WIDTH // 2 - 2)
        y0 = min(inset, HEIGHT // 2 - 2)
        x1 = max(x0 + 1, WIDTH - inset)
        y1 = max(y0 + 1, HEIGHT - inset)

        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=25,
            outline=(
                min(255, 35 + i * 7),
                min(255, 40 + i * 4),
                min(255, 100 + i * 5)
            ),
            width=3
        )

    draw.ellipse((455, 115, 645, 305), fill=(75, 40, 130))
    draw.ellipse((485, 145, 615, 275), fill=(110, 65, 175))
    draw.ellipse((35, 900, 220, 1085), fill=(35, 75, 135))

    small_font = load_font(20)
    title_font = load_font(42)
    subtitle_font = load_font(27)

    draw.text(
        (38, 45),
        "AI CONTENT FACTORY",
        font=small_font,
        fill=(190, 210, 255)
    )

    draw.rounded_rectangle(
        (32, 340, WIDTH - 32, 920),
        radius=32,
        fill=(23, 28, 65),
        outline=(120, 100, 210),
        width=4
    )

    title_lines = wrap_text(draw, topic, title_font, WIDTH - 115)
    y = 415

    for line in title_lines[:5]:
        draw.text(
            (58, y),
            line,
            font=title_font,
            fill=(255, 225, 150)
        )
        y += 62

    draw.line(
        (58, 745, WIDTH - 58, 745),
        fill=(100, 145, 240),
        width=3
    )

    subtitle_lines = wrap_text(
        draw, narration, subtitle_font, WIDTH - 115
    )

    y = 785
    for line in subtitle_lines[:3]:
        draw.text(
            (58, y),
            line,
            font=subtitle_font,
            fill=(245, 245, 255)
        )
        y += 42

    draw.text(
        (38, HEIGHT - 70),
        f"SCENE {scene_number}/{total_scenes}",
        font=small_font,
        fill=(190, 210, 255)
    )

    return image


def split_script(script, count=6):
    sentences = re.split(r"(?<=[।!?])\s+", script.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        sentences = [script.strip()]

    groups = [[] for _ in range(count)]

    for i, sentence in enumerate(sentences):
        index = min(i * count // len(sentences), count - 1)
        groups[index].append(sentence)

    result = [" ".join(group).strip() for group in groups]

    # Keep empty scenes meaningful if the script is very short.
    for i in range(len(result)):
        if not result[i]:
            result[i] = script.strip()

    return result


@app.get("/")
def health_check():
    return {
        "status": "Online",
        "engine": "AI Content Factory 3D Video Studio",
        "version": APP_VERSION
    }


@app.post("/generate-video")
def generate_content(req: ContentRequest):
    job_id = "JOB-" + uuid.uuid4().hex[:10].upper()
    work_dir = OUTPUT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path = work_dir / "narration.mp3"
    audio_download_path = OUTPUT_DIR / f"{job_id}.mp3"
    video_path = OUTPUT_DIR / f"{job_id}.mp4"

    audio_clip = None
    scene_clips = []
    video_clip = None

    try:
        topic = req.topic.strip()
        script = req.script.strip()

        if not topic:
            raise HTTPException(
                status_code=400,
                detail="Topic is required."
            )

        if not script:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Script is required. Generate the story in the app "
                    "and send it in the script field."
                )
            )

        if req.language != "hi":
            raise HTTPException(
                status_code=400,
                detail="This version supports Hindi narration only."
            )

        # Generate Hindi narration from the supplied complete script.
        gTTS(text=script, lang="hi", slow=False).save(str(audio_path))

        audio_clip = AudioFileClip(str(audio_path))
        audio_duration = audio_clip.duration
        target_duration = float(req.duration)

        if audio_duration > target_duration:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Your narration is {audio_duration:.1f} seconds long, "
                    f"but the selected video is {req.duration} seconds. "
                    "Shorten the script and retry."
                )
            )

        scene_texts = split_script(script, 6)
        scene_duration = target_duration / len(scene_texts)

        for index, narration_part in enumerate(scene_texts, start=1):
            image = make_scene(
                topic=topic,
                narration=narration_part,
                scene_number=index,
                total_scenes=len(scene_texts)
            )

            image_path = work_dir / f"scene_{index}.jpg"
            image.save(image_path, quality=92)

            clip = ImageClip(str(image_path)).with_duration(scene_duration)
            scene_clips.append(clip)

        video_clip = concatenate_videoclips(
            scene_clips,
            method="compose"
        )

        # Add the generated narration to the actual MP4.
        video_clip = video_clip.with_audio(audio_clip)

        video_clip.write_videofile(
            str(video_path),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=2,
            logger=None
        )

        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RuntimeError("The MP4 file was not created.")

        shutil.copy2(audio_path, audio_download_path)

        base_url = os.getenv(
            "PUBLIC_BASE_URL",
            "https://ai-content-backend-kgew.onrender.com"
        ).rstrip("/")

        return {
            "success": True,
            "job_id": job_id,
            "topic": topic,
            "duration_seconds": req.duration,
            "script_used": script,
            "video_url": f"{base_url}/downloads/{job_id}.mp4",
            "audio_url": f"{base_url}/downloads/{job_id}.mp3",
            "video_mime_type": "video/mp4",
            "audio_mime_type": "audio/mpeg",
            "storyboard": [
                {
                    "scene": i + 1,
                    "visual": "3D-style layered scene",
                    "narration": part
                }
                for i, part in enumerate(scene_texts)
            ],
            "status": "MP4 rendered successfully"
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video generation failed: {str(exc)}"
        )
    finally:
        if video_clip is not None:
            try:
                video_clip.close()
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
