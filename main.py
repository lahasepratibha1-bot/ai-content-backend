import os, re, uuid, shutil, threading, tempfile
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips

APP_VERSION = "11.0"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(Path(tempfile.gettempdir()) / "ai_content_factory")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT, FPS = 720, 1280, 24
MAX_SCRIPT_CHARS = int(os.getenv("MAX_SCRIPT_CHARS", "200000"))
# No fixed target duration is imposed. Practical limits are set by hosting resources.
app = FastAPI(title="Personal AI Video Studio", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.mount("/downloads", StaticFiles(directory=str(OUTPUT_DIR)), name="downloads")
JOBS, JOBS_LOCK = {}, threading.Lock()

class ContentRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    niche: str = Field(default="General", max_length=200)
    language: str = "hi"
    script: str = Field(min_length=1, max_length=MAX_SCRIPT_CHARS)
    duration: int | None = None
    aspect_ratio: str = "9:16"

def load_font(size):
    for path in ["/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
                 "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, size)
            except Exception: pass
    return ImageFont.load_default()

def wrap_text(draw, text, font, max_width):
    words=text.split(); lines=[]; current=""
    for word in words:
        candidate=(current+" "+word).strip()
        box=draw.textbbox((0,0),candidate,font=font)
        if box[2]-box[0] <= max_width: current=candidate
        else:
            if current: lines.append(current)
            current=word
    if current: lines.append(current)
    return lines

def make_scene(topic, narration, n, total, size):
    w,h=size
    im=Image.new("RGB",(w,h),(12,15,38)); d=ImageDraw.Draw(im)
    for i in range(12):
        inset=i*max(3,w//55)
        d.rounded_rectangle((inset,inset,w-inset,h-inset),radius=max(5,w//30),
                            outline=(min(255,35+i*12),min(255,45+i*8),min(255,110+i*9)),width=max(1,w//360))
    d.ellipse((int(w*.66),int(h*.08),int(w*.9),int(h*.25)),fill=(75,40,130))
    d.ellipse((int(w*.7),int(h*.11),int(w*.86),int(h*.22)),fill=(110,65,175))
    small,title,sub=load_font(max(12,w//36)),load_font(max(20,w//17)),load_font(max(16,w//27))
    d.text((int(w*.05),int(h*.04)),"PERSONAL AI VIDEO STUDIO",font=small,fill=(190,210,255))
    d.rounded_rectangle((int(w*.045),int(h*.28),int(w*.955),int(h*.76)),radius=max(10,w//35),fill=(23,28,65),outline=(120,100,210),width=max(2,w//180))
    y=int(h*.34)
    for line in wrap_text(d,topic,title,int(w*.82))[:5]:
        d.text((int(w*.08),y),line,font=title,fill=(255,225,150));y+=int(h*.055)
    d.line((int(w*.08),int(h*.61),int(w*.92),int(h*.61)),fill=(100,145,240),width=max(2,w//240))
    y=int(h*.64)
    for line in wrap_text(d,narration,sub,int(w*.82))[:4]:
        d.text((int(w*.08),y),line,font=sub,fill=(245,245,255));y+=int(h*.04)
    d.text((int(w*.05),int(h*.94)),f"SCENE {n}/{total}",font=small,fill=(190,210,255))
    return im

def split_sentences(script):
    parts=[x.strip() for x in re.split(r"(?<=[.!?।])\s+",script.strip()) if x.strip()]
    return parts or [script.strip()]

def split_for_tts(text, max_chars=2800):
    # Split at sentence and word boundaries so long scripts do not become one huge TTS request.
    sentences = split_sentences(text)
    pieces = []
    for sentence in sentences:
        remaining = sentence.strip()
        while len(remaining) > max_chars:
            cut = remaining.rfind(" ", 0, max_chars)
            if cut < 1:
                cut = max_chars
            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            pieces.append(remaining)
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = (current + " " + piece).strip()
    if current:
        chunks.append(current)
    return chunks or [text[:max_chars]]

def update_job(job_id, **values):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values);JOBS[job_id]["updated_at"]=datetime.now(timezone.utc).isoformat()

def render_job(job_id, topic, script, language, aspect_ratio):
    work=OUTPUT_DIR/(job_id+"_work");work.mkdir(parents=True,exist_ok=True)
    audio_clips=[];scene_clips=[];final_clip=None;full_audio=None
    video_path=OUTPUT_DIR/f"{job_id}.mp4";audio_path=work/"narration.mp3"
    try:
        update_job(job_id,status="processing",progress=3,message="Generating narration in chunks")
        chunks=split_for_tts(script)
        audio_paths=[]
        for i,chunk in enumerate(chunks,1):
            p=work/f"audio_{i:04d}.mp3"
            gTTS(text=chunk,lang=language,slow=False).save(str(p));audio_paths.append(p)
            update_job(job_id,progress=min(25,3+int(22*i/len(chunks))),message=f"Generated narration chunk {i}/{len(chunks)}")
        for p in audio_paths: audio_clips.append(AudioFileClip(str(p)))
        full_audio=concatenate_audioclips(audio_clips)
        audio_duration=float(full_audio.duration or 0)
        if audio_duration<=0: raise RuntimeError("Could not detect generated narration duration.")
        if aspect_ratio=="16:9": size=(1280,720)
        elif aspect_ratio=="1:1": size=(900,900)
        else: size=(720,1280)
        sentences=split_sentences(script)
        # Keep scene cards manageable; every sentence is represented, and scene timing follows audio duration.
        groups=[];current=[]
        for sent in sentences:
            current.append(sent)
            if len(" ".join(current))>=220 or len(current)>=3:
                groups.append(" ".join(current));current=[]
        if current: groups.append(" ".join(current))
        groups=groups or [script]
        scene_duration=audio_duration/len(groups)
        update_job(job_id,progress=28,message=f"Preparing {len(groups)} scenes",duration_seconds=round(audio_duration,2),total_scenes=len(groups))
        for i,part in enumerate(groups,1):
            img=make_scene(topic,part,i,len(groups),size);ip=work/f"scene_{i:04d}.jpg";img.save(ip,quality=90)
            scene_clips.append(ImageClip(str(ip)).with_duration(scene_duration))
            if i%5==0 or i==len(groups):
                update_job(job_id,progress=28+int(40*i/len(groups)),message=f"Prepared scene {i}/{len(groups)}")
        update_job(job_id,progress=70,message="Rendering MP4 with narration audio")
        final_clip=concatenate_videoclips(scene_clips,method="compose").with_audio(full_audio)
        final_clip.write_videofile(str(video_path),fps=FPS,codec="libx264",audio_codec="aac",preset="ultrafast",threads=2,logger=None)
        if not video_path.exists() or video_path.stat().st_size==0: raise RuntimeError("MP4 output was not created.")
        # Validate a muxed audio track via MoviePy's output duration/audio presence.
        base=os.getenv("PUBLIC_BASE_URL","https://ai-content-backend-kgew.onrender.com").rstrip("/")
        update_job(job_id,status="completed",progress=100,message="MP4 rendered with narration audio",topic=topic,
                   duration_seconds=round(audio_duration,2),video_url=f"{base}/downloads/{job_id}.mp4",
                   audio_url=None,total_scenes=len(groups))
    except Exception as e:
        update_job(job_id,status="failed",progress=100,message=str(e),error=str(e))
    finally:
        if final_clip:
            try: final_clip.close()
            except Exception: pass
        if full_audio:
            try: full_audio.close()
            except Exception: pass
        for c in audio_clips:
            try: c.close()
            except Exception: pass
        for c in scene_clips:
            try: c.close()
            except Exception: pass
        shutil.rmtree(work,ignore_errors=True)

@app.get("/")
def health():
    return {"status":"Online","engine":"Personal AI Video Studio","version":APP_VERSION,"mode":"server-rendered narrated MP4"}

@app.post("/generate-video")
def generate_video(req: ContentRequest):
    topic=req.topic.strip();script=req.script.strip()
    if not topic: raise HTTPException(status_code=400,detail="Topic is required.")
    if not script: raise HTTPException(status_code=400,detail="Script is required.")
    if len(script)>MAX_SCRIPT_CHARS: raise HTTPException(status_code=413,detail=f"Script is too large for configured limit ({MAX_SCRIPT_CHARS} characters).")
    if req.language!="hi": raise HTTPException(status_code=400,detail="Current backend supports Hindi narration only.")
    if req.aspect_ratio not in ("9:16","16:9","1:1"): raise HTTPException(status_code=400,detail="Unsupported aspect ratio.")
    jid="JOB-"+uuid.uuid4().hex[:10].upper();now=datetime.now(timezone.utc).isoformat()
    with JOBS_LOCK:
        JOBS[jid]={"success":True,"job_id":jid,"topic":topic,"status":"queued","progress":0,"message":"Video job queued","created_at":now,"updated_at":now}
    threading.Thread(target=render_job,args=(jid,topic,script,req.language,req.aspect_ratio),daemon=True).start()
    return {"success":True,"job_id":jid,"status":"queued","message":"Video rendering started."}

@app.get("/jobs/{job_id}")
def get_job(job_id:str):
    with JOBS_LOCK:
        job=JOBS.get(job_id)
        if not job: raise HTTPException(status_code=404,detail="Job not found. The service may have restarted; submit the job again.")
        return dict(job)
