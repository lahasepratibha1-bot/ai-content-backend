import os
import tempfile
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

app = FastAPI(title="AI Content Factory Autonomous Video Studio v4.2", version="4.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoRequest(BaseModel):
    topic: str
    niche: str = "Emotional Stories"
    style: str = "3D Animation"
    language: str = "hi"

@app.get("/")
def health_check():
    return {"status": "Online", "engine": "Autonomous Cloud Video Rendering Engine v4.2"}

@app.post("/generate-video")
async def generate_video(req: VideoRequest):
    temp_files = []
    try:
        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required.")
        
        narration_text = (
            f"Namaskar doston. Aaj ka hamara vishesh vishay hai: {topic}. "
            f"Is kahani ke madhyam se hum yeh jaanenge ki kaise hamare jeevan mein "
            f"chote-chote badlav aur rishte sabse bade maayne rakhte hain. "
            f"Kripya is safar mein ant tak bane rahiye."
        )
        
        # Generate Audio using gTTS
        tts = gTTS(text=narration_text, lang=req.language, slow=False)
        audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
        os.close(audio_fd)
        tts.save(audio_path)
        temp_files.append(audio_path)
        
        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration
        
        visual_prompts = [
            f"{topic} 3d pixar style cute character emotional expression cinematic lighting masterpiece",
            f"{topic} dramatic cinematic scene 3d animation digital art high detail vibrant colors",
            f"{topic} touching emotional storytelling 3d render atmospheric lighting depth of field"
        ]
        
        clip_duration = max(2.5, total_duration / len(visual_prompts))
        image_clips = []
        
        for idx, p in enumerate(visual_prompts):
            img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(p)}?width=1080&height=1920&nologo=true&seed={1500+idx}"
            
            try:
                img_response = requests.get(img_url, timeout=20)
                if img_response.status_code == 200:
                    img_fd, img_path = tempfile.mkstemp(suffix=".jpg")
                    os.close(img_fd)
                    with open(img_path, "wb") as f:
                        f.write(img_response.content)
                    temp_files.append(img_path)
                    
                    img_clip = ImageClip(img_path).set_duration(clip_duration)
                    image_clips.append(img_clip)
            except Exception:
                pass
                
        if not image_clips:
            audio_clip.close()
            raise HTTPException(status_code=500, detail="Failed to generate visual assets.")
            
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

        job_id = f"MP4-JOB-{os.urandom(4).hex()}"
        
        return {
            "success": True,
            "job_id": job_id,
            "topic": topic,
            "style": req.style,
            "duration_seconds": round(total_duration, 2),
            "video_url": f"data:video/mp4;base64,{video_base64}",
            "status": "Multi-Scene MP4 Video Rendered Successfully"
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
