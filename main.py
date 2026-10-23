import os
import tempfile
import base64
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import edge_tts
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

app = FastAPI(title="AI Content Factory Professional Studio v4.1", version="4.1")

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
    return {"status": "Online", "engine": "Professional Neural Video Engine v4.1"}

async def generate_natural_voice(text: str, output_path: str):
    # Using Microsoft Neural Professional Hindi Voice (SwaraNeural)
    communicate = edge_tts.Communicate(text, "hi-IN-SwaraNeural")
    await communicate.save(output_path)

@app.post("/generate-video")
async def generate_video(req: VideoRequest):
    temp_files = []
    try:
        topic = req.topic.strip()
        if not topic:
            raise HTTPException(status_code=400, detail="Topic is required.")
        
        # Professional Script Generation
        script_text = (
            f"Namaskar doston. Aaj ka hamara vishay hai: {topic}. "
            f"Zindagi ke safar mein har mod par kuch aisi baatein hoti hain jo hamari rooh ko choo leti hain. "
            f"Aaiye is khubsurat kahani ko gehraai se samajhte hain aur iske asli sach ko jaante hain."
        )
        
        # 1. Generate Natural Neural Voiceover Audio
        audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
        os.close(audio_fd)
        temp_files.append(audio_path)
        
        await generate_natural_voice(script_text, audio_path)
        
        audio_clip = AudioFileClip(audio_path)
        total_duration = audio_clip.duration
        
        # 2. Multiple Cinematic 3D Visual Prompts for Scene Transitions
        prompts = [
            f"{topic} 3d pixar style cute character emotional expression cinematic lighting masterpiece",
            f"{topic} dramatic cinematic scene 3d animation digital art high detail vibrant colors",
            f"{topic} touching emotional storytelling 3d render atmospheric lighting depth of field"
        ]
        
        clip_duration = max(2.5, total_duration / len(prompts))
        image_clips = []
        
        for idx, p in enumerate(prompts):
            img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(p)}?width=1080&height=1920&nologo=true&seed={1500+idx}"
            
            try:
                img_response = requests.get(img_url, timeout=20)
                if img_response.status_code == 200:
                    img_fd, img_path = tempfile.mkstemp(suffix=".jpg")
                    os.close(img_fd)
                    with open(img_path, "wb") as f:
                        f.write(img_response.content)
                    temp_files.append(img_path)
                    
                    # Create Image Clip with smooth Ken Burns Zoom Effect
                    img_clip = ImageClip(img_path).set_duration(clip_duration)
                    # Apply smooth resize/zoom effect for professional motion
                    img_clip = img_clip.resize(lambda t: 1.0 + 0.04 * t) 
                    image_clips.append(img_clip)
            except Exception:
                pass
                
        if not image_clips:
            audio_clip.close()
            raise HTTPException(status_code=500, detail="Failed to generate visual assets.")
            
        # 3. Stitch Scenes and Neural Audio into Final MP4 Video
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
        
        # 4. Convert MP4 to Base64 for web player
        with open(vid_path, "rb") as video_file:
            video_bytes = video_file.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        job_id = f"PRO-MP4-{os.urandom(4).hex()}"
        
        return {
            "success": True,
            "job_id": job_id,
            "topic": topic,
            "style": req.style,
            "duration_seconds": round(total_duration, 2),
            "video_url": f"data:video/mp4;base64,{video_base64}",
            "status": "Professional Neural Video Rendered Successfully"
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
