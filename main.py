import os
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

app = FastAPI(title="AI Content Factory Autonomous Video Studio v3.5", version="3.5")

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
    return {"status": "Online", "engine": "Autonomous Cloud Video Rendering Engine v3.5"}

@app.post("/generate-video")
async def generate_video(req: VideoRequest):
    temp_files = []
    try:
        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required.")
        
        # 1. Generate Script / Narration text
        narration_text = f"Aaj ki kahani ka vishay hai: {topic}. Zindagi mein har mushkil ke piche ek seekh hoti hai jo hamein mazboot banati hai."
        
        # 2. Generate Audio using gTTS
        tts = gTTS(text=narration_text, lang=req.language, slow=False)
        audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
        os.close(audio_fd)
        tts.save(audio_path)
        temp_files.append(audio_path)
        
        # Load audio to get duration
        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration
        
        # 3. Generate 3D Visual Assets (Images)
        prompts = [
            f"{topic} 3d pixar style cute character emotional expression cinematic lighting",
            f"{topic} dramatic scene 3d animation digital art masterpiece",
            f"{topic} touching moment emotional storytelling 3d render"
        ]
        
        clip_duration = max(2.0, total_duration / len(prompts))
        image_clips = []
        
        for idx, p in enumerate(prompts):
            img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(p)}?width=1080&height=1920&nologo=true&seed={700+idx}"
            
            # Download image locally
            img_response = requests.get(img_url, timeout=15)
            if img_response.status_code == 200:
                img_fd, img_path = tempfile.mkstemp(suffix=".jpg")
                os.close(img_fd)
                with open(img_path, "wb") as f:
                    f.write(img_response.content)
                temp_files.append(img_path)
                
                # Create MoviePy Image Clip
                img_clip = ImageClip(img_path).set_duration(clip_duration)
                image_clips.append(img_clip)
                
        if not image_clips:
            raise HTTPException(status_code=500, detail="Failed to generate visual assets.")
            
        # 4. Stitch Images together into Video
        video_sequence = concatenate_videoclips(image_clips, method="compose")
        final_video = video_sequence.set_audio(audio_clip)
        
        # Output MP4 path
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
        
        # Clean up clips
        audio_clip.close()
        final_video.close()

        # Job ID & Response (In production, video can be uploaded to Cloud Storage / Google Drive. Here we return success metrics and cloud job details)
        job_id = f"MP4-JOB-{os.urandom(4).hex()}"
        
        return {
            "success": True,
            "job_id": job_id,
            "topic": topic,
            "style": req.style,
            "duration_seconds": round(total_duration, 2),
            "status": "Real MP4 Video Rendered Successfully on Cloud Server"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temporary files safely
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
