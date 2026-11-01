import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gtts import gTTS

app = FastAPI(
    title="Personal AI Video Studio",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


class AudioRequest(BaseModel):
    script: str = Field(min_length=1, max_length=10000)
    language: str = "hi"
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0)


@app.get("/")
def home():
    return {
        "success": True,
        "engine": "Personal AI Video Studio",
        "status": "online",
        "version": "1.0.0"
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.post("/api/generate-audio")
def generate_audio(data: AudioRequest):
    supported_languages = ["hi", "mr", "en"]

    if data.language not in supported_languages:
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
