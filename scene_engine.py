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
    visual_style: str = "cinematic realistic"
    target_duration: int = Field(default=60, ge=5, le=3600)
def split_sentences(script):
    return [x.strip() for x in re.split(r'(?<=[.!?।])\s+|\n+', script.strip()) if x.strip()]
def detect_emotion(t):
    s=t.lower()
    if any(x in s for x in ["खुशी के आंसू","खुशी के आँसू","खुशी","मुस्कुर","happy","smile"]): return "happy"
    if any(x in s for x in ["रोई","रोया","दुख","उदास","आंसू","आँसू","sad","cry"]): return "sad"
    if any(x in s for x in ["गुस्सा","चिल्ला","नाराज","angry"]): return "angry"
    if any(x in s for x in ["प्यार","मां","माँ","बेटी","गले लगा","love","mother","daughter"]): return "love"
    return "natural"
def detect_action(t):
    s=t.lower()
    rules=[("waking_up",["उठी","उठा","जागी","जागा","उठती"]),("hugging",["गले लगा","लिपट","hug"]),("cooking",["खाना बनाती","खाना बनाता","बनाती","बनाता","cooking"]),("crying",["रोई","रोया","रोने","crying"]),("smiling",["मुस्कुर","हंसी","smile"]),("walking",["चलती","चला","गई","गया","walking"]),("speaking",["बोली","बोला","कहा","पूछा"])]
    for action, words in rules:
        if any(w in s for w in words): return action
    return "natural_action"
@router.post("/plan")
def create_scene_plan(data: SceneRequest):
    if data.language not in ["hi","mr","en"]: raise HTTPException(400,"Supported languages: hi, mr, en")
    if data.video_format not in ["shorts","landscape"]: raise HTTPException(400,"Format must be shorts or landscape")
    texts=split_sentences(data.script)
    if not texts: raise HTTPException(400,"No scenes could be created")
    jid=str(uuid.uuid4()); scenes=[]
    for i, text in enumerate(texts,1):
        action, emotion=detect_action(text), detect_emotion(text)
        ratio="9:16 vertical" if data.video_format=="shorts" else "16:9 landscape"
        prompt=(f"{data.visual_style} cinematic scene, {ratio}. Scene {i}. Story action: {text}. Characters: {data.character_description}. Action: {action}. Emotion: {emotion}. Keep character face, age, hairstyle and clothing consistent. Detailed environment, natural anatomy, cinematic light. No text, no watermark.")
        scenes.append({"scene_id":str(uuid.uuid4()),"scene_number":i,"scene_text":text,"duration_seconds":max(3,min(10,round(len(text.split())/2.5))),"characters":data.character_description,"action":action,"emotion":emotion,"location":"To be identified by AI","camera_direction":"Cinematic medium shot","dialogue_or_voiceover":text,"image_prompt":prompt,"motion_prompt":f"Animate {action} with {emotion} emotion. Context: {text}. Natural body movement, stable camera.","image_status":"not_generated","video_status":"not_generated","image_url":None,"video_url":None})
    job={"job_id":jid,"status":"planned","planner_type":"rule_based_v1","language":data.language,"video_format":data.video_format,"visual_style":data.visual_style,"target_duration":data.target_duration,"character_description":data.character_description,"total_scenes":len(scenes),"scenes":scenes,"created_at":datetime.now(timezone.utc).isoformat()}
    SCENE_JOBS[jid]=job
    return {"success":True,"message":"Scene plan created","job":job}
@router.get("/jobs")
def get_scene_jobs(): return {"success":True,"total_jobs":len(SCENE_JOBS),"jobs":list(SCENE_JOBS.values())}
@router.get("/jobs/{job_id}")
def get_scene_job(job_id:str):
    job=SCENE_JOBS.get(job_id)
    if not job: raise HTTPException(404,"Scene job not found")
    return {"success":True,"job":job}
