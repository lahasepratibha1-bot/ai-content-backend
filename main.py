import os
import tempfile
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gtts import gTTS
from moviepy.editor import ColorClip, AudioFileClip, concatenate_videoclips

app = FastAPI(title="AI Content Factory Stable Studio v4.6", version="4.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    topic: str
    niche: str = "General"
    style: str = "3D Animation"
    language: str = "hi"

@app.get("/")
def health_check():
    return {"status": "Online", "engine": "Stable Cloud Video Engine v4.6"}

@app.post("/generate-video")
async def generate_video(req: VideoRequest):
    temp_files = []
    try:
        topic = req.topic.strip()
        if not topic:
            topic = "Jeevan Ka Sach"
        
        narration_text = (
            f"Namaskar doston. Aaj ka hamara vishesh vishay hai: {topic}. "
            f"Is kahani ke madhyam se hum yeh jaanenge ki kaise hamare jeevan mein "
            f"chote-chote vichar aur badlav sabse bade maayne rakhte hain. "
            f"Kripya is safar mein ant tak bane rahiye."
        )
        
        # 1. Generate Audio using gTTS
        tts = gTTS(text=narration_text, lang=req.language, slow=False)
        audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
        os.close(audio_fd)
        tts.save(audio_path)
        temp_files.append(audio_path)
        
        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration
        
        # 2. Create stable background video clips to prevent server timeouts
        clip_duration = max(3.0, total_duration / 3)
        bg_colors = [
            [15, 23, 42],   # Dark Slate Blue
            [30, 41, 59],   # Lighter Slate
            [17, 24, 39]    # Deep Indigo
        ]
        
        image_clips = []
        for color in bg_colors:
            c = ColorClip(size=(1080, 1920), color=color).set_duration(clip_duration)
            image_clips.append(c)
                
        video_sequence = concatenate_videoclips(image_clips, method="compose")
        final_video = video_sequence.set_audio(audio_clip)
        
        vid_fd, vid_path = tempfile.mkstemp(suffix=".mp4")
        os.close(vid_fd)
        temp_files.append(vid_path)
        
        final_video.write_videofile(
            vid_path, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac", 
            preset="ultrafast", 
            logger=None
        )
        
        audio_clip.close()
        final_video.close()
        
        with open(vid_path, "rb") as video_file:
            video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        job_id = f"STABLE-JOB-{os.urandom(4).hex()}"
        
        return {
            "success": True,
            "job_id": job_id,
            "topic": topic,
            "style": req.style,
            "duration_seconds": round(total_duration, 2),
            "video_url": f"data:video/mp4;base64,{video_base64}",
            "status": "Stable Video Rendered Successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
