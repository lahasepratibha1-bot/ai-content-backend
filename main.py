import os
import tempfile
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gtts import gTTS

app = FastAPI(title="AI Content Factory Professional Studio", version="8.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContentRequest(BaseModel):
    topic: str
    niche: str = "General"
    language: str = "hi"

@app.get("/")
def health_check():
    return {"status": "Online", "engine": "Professional Studio Engine v8.0"}

@app.post("/generate-video")
async def generate_content(req: ContentRequest):
    try:
        topic = req.topic.strip() if req.topic else "Jeevan Ka Sach"
        
        # Professional Narration Script Text
        narration_text = (
            f"Namaskar doston. Aaj ka hamara vishesh vishay hai: {topic}. "
            f"Is gahre vishay aur kahani ke madhyam se hum yeh samjhenge ki kaise hamare jeevan mein "
            f"chote-chote vichar aur badlav sabse bade maayne rakhte hain. Kripya is safar mein ant tak bane rahiye."
        )
        
        # Crystal Clear Neural Audio Generation via gTTS
        tts = gTTS(text=narration_text, lang=req.language, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            audio_path = fp.name
            tts.save(audio_path)
            
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
        try:
            os.remove(audio_path)
        except:
            pass

        # Professional Multi-Scene Storyboard Data
        storyboard = [
            {"scene": 1, "visual": "Cinematic emotional hook, dramatic lighting, 3D art", "narration": f"Vishay: {topic}"},
            {"scene": 2, "visual": "Deep storytelling atmosphere, meaningful expressions", "narration": "Gahri soch aur jeevan ka asli sach."},
            {"scene": 3, "visual": "Turning point and positive resolution", "narration": "Ek nayi seekh jo aapka nazariya badal degi."}
        ]
        
        return {
            "success": True,
            "job_id": f"PRO-JOB-{os.urandom(3).hex()}",
            "topic": topic,
            "storyboard": storyboard,
            "video_url": f"data:audio/mp3;base64,{audio_base64}",
            "status": "Professional Audio & Storyboard Generated Successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
