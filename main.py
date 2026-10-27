import os
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gtts import gTTS

app = FastAPI(title="Personal AI Video Studio - Local Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ScriptRequest(BaseModel):
    topic: str
    language: str = "hi"

class RenderRequest(BaseModel):
    script: str
    aspect_ratio: str = "9:16"
    voice_speed: float = 1.1

@app.post("/api/generate-audio")
def generate_audio(data: RenderRequest):
    try:
        # Step 8 & 9: Local TTS generation & actual duration measurement
        tts = gTTS(text=data.script, lang='hi', slow=False)
        audio_path = os.path.join(OUTPUT_DIR, "narration.mp3")
        tts.save(audio_path)
        
        return {
            "success": True,
            "audio_url": f"/outputs/narration.mp3",
            "message": "Audio narration generated successfully locally."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
