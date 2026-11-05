import re, uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/scenes", tags=["AI Scene Engine"])
SCENE_JOBS = {}

class SceneRequest(BaseModel):
    script: str = Field(min_length=1, max_length=50000)
    language: str = "hi"
    video_format: str = "shorts"
    character_description: str = ""
    visual_style: str = "photorealistic Indian cinema still, cinematic natural lighting"
    target_duration: int = Field(default=60, ge=5, le=3600)

def split_sentences(script):
    # Keep scene count manageable: combine very short sentences; split long lines by punctuation.
    chunks = [x.strip() for x in re.split(r'(?<=[.!?।])\s+|\n+', script.strip()) if x.strip()]
    scenes, current = [], ""
    for chunk in chunks:
        if len((current + " " + chunk).split()) <= 18 or not current:
            current = (current + " " + chunk).strip()
        else:
            scenes.append(current); current = chunk
    if current: scenes.append(current)
    return scenes or [script.strip()]

def detect_emotion(t):
    s=t.lower()
    if any(x in s for x in ["खुशी के आंसू","खुशी के आँसू","खुशी","मुस्कुर","happy","smile"]): return "warm happiness"
    if any(x in s for x in ["रोई","रोया","दुख","उदास","आंसू","आँसू","sad","cry"]): return "quiet sadness"
    if any(x in s for x in ["गुस्सा","चिल्ला","नाराज","angry"]): return "restrained anger"
    if any(x in s for x in ["प्यार","मां","माँ","बेटी","गले लगा","love","mother","daughter"]): return "tender affection"
    return "natural everyday emotion"

def detect_action(t):
    s=t.lower()
    rules=[("waking up",["उठी","उठा","जागी","जागा","उठती"]),("embracing",["गले लगा","लिपट","hug"]),("cooking",["खाना बनाती","खाना बनाता","बनाती","बनाता","cooking"]),("crying",["रोई","रोया","रोने","crying"]),("smiling",["मुस्कुर","हंसी","smile"]),("walking",["चलती","चला","गई","गया","walking"]),("speaking",["बोली","बोला","कहा","पूछा"])]
    for action, words in rules:
        if any(w in s for w in words): return action
    return "a natural candid moment"

@router.post("/plan")
def create_scene_plan(data: SceneRequest):
    if data.language not in ["hi","mr","en"]: raise HTTPException(400,"Supported languages: hi, mr, en")
    if data.video_format not in ["shorts","landscape"]: raise HTTPException(400,"Format must be shorts or landscape")
    texts=split_sentences(data.script)
    jid=str(uuid.uuid4()); scenes=[]
    ratio="vertical 9:16 portrait composition" if data.video_format=="shorts" else "horizontal 16:9 composition"
    character = data.character_description.strip() or "consistent everyday Indian characters appropriate to the story; realistic age, face, hairstyle and clothing"
    for i, text in enumerate(texts,1):
        action, emotion=detect_action(text), detect_emotion(text)
        prompt=(f"{data.visual_style}, {ratio}. A single coherent cinematic film still for scene {i}. "
                f"Show this exact story moment: {text}. Characters and continuity: {character}. "
                f"Visible action: {action}. Facial emotion: {emotion}. "
                "Photorealistic human faces, believable Indian setting and details, natural skin texture, "
                "realistic hands, professional cinematography, expressive composition, detailed environment, "
                "soft depth of field, natural color grading. No words, no letters, no subtitles, no logos, no watermark.")
        scenes.append({"scene_id":str(uuid.uuid4()),"scene_number":i,"scene_text":text,
            "duration_seconds":max(3,min(8,round(len(text.split())/2.5))),"characters":character,
            "action":action,"emotion":emotion,"location":"story setting","camera_direction":"cinematic medium shot",
            "dialogue_or_voiceover":text,"image_prompt":prompt,
            "motion_prompt":f"Subtle cinematic camera movement for {action}; preserve natural expression and anatomy.",
            "image_status":"not_generated","video_status":"not_generated","image_url":None,"video_url":None})
    job={"job_id":jid,"status":"planned","planner_type":"rule_based_v3","language":data.language,
         "video_format":data.video_format,"visual_style":data.visual_style,"target_duration":data.target_duration,
         "character_description":character,"total_scenes":len(scenes),"scenes":scenes,
         "created_at":datetime.now(timezone.utc).isoformat()}
    SCENE_JOBS[jid]=job
    return {"success":True,"message":"Scene plan created","job":job}

@router.get("/jobs")
def get_scene_jobs(): return {"success":True,"total_jobs":len(SCENE_JOBS),"jobs":list(SCENE_JOBS.values())}

@router.get("/jobs/{job_id}")
def get_scene_job(job_id:str):
    job=SCENE_JOBS.get(job_id)
    if not job: raise HTTPException(404,"Scene job not found")
    return {"success":True,"job":job}
