
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
from moviepy import (
    ImageClip,
    AudioFileClip,
    concatenate_videoclips,
)

APP_VERSION = "9.0"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "ai_content_factory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 720
HEIGHT = 1280
VIDEO_SECONDS = 60
FPS = 24

app = FastAPI(
    title="AI Content Factory 3D Video Studio",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/downloads",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="downloads",
)


class ContentRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=250)
    niche: str = Field(default="General", max_length=150)
    language: str = "hi"
    script: str = Field(default="", max_length=12000)
    duration: int = Field(default=60, ge=10, le=60)


def find_font():
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]

    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, 38)
            except Exception:
                pass

    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
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
    # Create a cinematic, layered 3D-look graphic scene.
    image = Image.new("RGB", (WIDTH, HEIGHT), (12, 15, 38))
    draw = ImageDraw.Draw(image)

    # Layered atmospheric backgrounds and glowing circles.
    for i in range(18):
        inset = i * 24
        color = (
            max(12, 40 - i),
            max(12, 35 - i),
            max(35, 95 - i * 3),
        )
        draw.rounded_rectangle(
            (inset, inset, WIDTH - inset, HEIGHT - inset),
            radius=40,
            outline=color,
            width=4,
        )

    draw.ellipse((430, 110, 650, 330), fill=(75, 40, 130))
    draw.ellipse((465, 145, 615, 295), fill=(110, 65, 175))
    draw.ellipse((35, 760, 260, 985), fill=(35, 75, 135))

    font = find_font()
    small_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22
    ) if os.path.exists(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ) else ImageFont.load_default()

    draw.text(
        (45, 48),
        "AI CONTENT FACTORY  |  3D STORY",
        font=small_font,
        fill=(190, 210, 255),
    )

    draw.rounded_rectangle(
        (35, 350, WIDTH - 35, 900),
        radius=35,
        fill=(23, 28, 65),
        outline=(120, 100, 210),
        width=4,
    )

    title = topic.strip()
    title_lines = wrap_text(draw, title, font, WIDTH - 130)

    y = 420
    for line in title_lines[:5]:
        draw.text((65, y), line, font=font, fill=(255, 225, 150))
        y += 60

    draw.line((65, 750, WIDTH - 65, 750), fill=(100, 145, 240), width=3)

    subtitle_font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25
    ) if os.path.exists(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ) else ImageFont.load_default()

    lines = wrap_text(draw, narration, subtitle_font, WIDTH - 120)
    y = 785

    for line in lines[:3]:
        draw.text((60, y), line, font=subtitle_font, fill=(245, 245, 255))
        y += 38

    draw.text(
        (45, HEIGHT - 75),
        f"SCENE {scene_number}/{total_scenes}",
        font=small_font,
        fill=(190, 210, 255),
    )

    return image


def split_script(script, count=6):
    # Split by sentences first, then distribute sentences across scenes.
    sentences = re.split(r"(?<=[।!?])\s+", script.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        sentences = [script.strip()]

    groups = [[] for _ in range(count)]

    for i, sentence in enumerate(sentences):
        groups[min(i * count // len(sentences), count - 1)].append(sentence)

    result = []
    for group in groups:
        text = " ".join(group).strip()
        result.append(text or script.strip())

    return result


@app.get("/")
def health_check():
    return {
        "status": "Online",
        "engine": "AI Content Factory 3D Video Studio",
        "version": APP_VERSION,
    }


@app.post("/generate-video")
def generate_content(req: ContentRequest):
    job_id = f"JOB-{uuid.uuid4().hex[:10].upper()}"
    work_dir = OUTPUT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path = work_dir / "narration.mp3"
    video_path = OUTPUT_DIR / f"{job_id}.mp4"

    audio_clip = None
    scene_clips = []

    try:
        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required.")

        script = req.script.strip()

        # Never pretend a topic-only generic narration is the full script.
        if not script:
            script = (
                f"{topic}। यह कहानी हमें जीवन की एक महत्वपूर्ण सीख देती है। "
                "कठिन समय में भी उम्मीद नहीं छोड़नी चाहिए। "
                "अपने रिश्तों की कद्र करें और दूसरों के दर्द को समझें। "
                "कभी-कभी एक छोटा सा अच्छा काम किसी की पूरी जिंदगी बदल सकता है। "
                "अगर यह कहानी आपके दिल को छू गई, तो इसे दूसरों के साथ जरूर साझा करें।"
            )

        if req.language != "hi":
            raise HTTPException(
                status_code=400,
                detail="This version currently supports Hindi narration only. Use language='hi'.",
            )

        # Generate actual speech from the supplied script.
        tts = gTTS(text=script, lang="hi", slow=False)
        tts.save(str(audio_path))

        audio_clip = AudioFileClip(str(audio_path))
        audio_duration = audio_clip.duration

        target_duration = float(req.duration)

        # Do not silently cut off the user's narration.
        if audio_duration > target_duration + 0.5:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Script narration is {audio_duration:.1f} seconds, "
                    f"longer than the selected {req.duration}-second video. "
                    "Please shorten the script and try again."
                ),
            )

        # If narration is shorter than 60 seconds, the final scene holds
        # until the target duration. The narration itself is not altered.
        scenes = split_script(script, count=6)
        scene_duration = target_duration / len(scenes)

        for index, narration_part in enumerate(scenes, start=1):
            image = make_scene(
                topic=topic,
                narration=narration_part,
                scene_number=index,
                total_scenes=len(scenes),
            )

            image_path = work_dir / f"scene_{index}.jpg"
            image.save(image_path, quality=92)

            clip = ImageClip(str(image_path)).with_duration(scene_duration)
            scene_clips.append(clip)

        video = concatenate_videoclips(scene_clips, method="compose")

        # Audio is placed on the rendered MP4.
        # The video remains 60 seconds, even if speech is shorter.
        video = video.with_audio(audio_clip)

        video.write_videofile(
            str(video_path),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            threads=2,
            logger=None,
        )

        video.close()
        audio_clip.close()
        audio_clip = None

        for clip in scene_clips:
            clip.close()
        scene_clips = []

        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RuntimeError("MP4 file was not created.")

        base_url = os.getenv(
            "PUBLIC_BASE_URL",
            "https://ai-content-backend-kgew.onrender.com",
        ).rstrip("/")

        # Separate real MP4 and MP3 download URLs.
        audio_download_path = OUTPUT_DIR / f"{job_id}.mp3"
        shutil.copy2(audio_path, audio_download_path)

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
                    "visual": "3D-style layered graphic scene with motion-ready design",
                    "narration": part,
                }
                for i, part in enumerate(scenes)
            ],
            "status": "MP4 rendered successfully",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video generation failed: {str(exc)}",
        )
    finally:
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
