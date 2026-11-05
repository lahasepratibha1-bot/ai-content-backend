import base64, os
from pathlib import Path
from typing import Optional
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scene_engine import SCENE_JOBS

router = APIRouter(prefix="/api/images", tags=["AI Image Engine"])
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
WORKER_URL = os.getenv("IMAGE_WORKER_URL", "").rstrip("/")
WORKER_TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "")
TIMEOUT = int(os.getenv("IMAGE_WORKER_TIMEOUT", "300"))

class ImageRequest(BaseModel):
    width: int = Field(default=576, ge=256, le=1024)
    height: int = Field(default=1024, ge=256, le=1536)
    strength: float = Field(default=0.62, ge=0.2, le=0.9)
    generate_character_reference: bool = True
    reference_image_base64: Optional[str] = None

def worker_health():
    if not WORKER_URL:
        return {"success":False,"configured":False,"error":"IMAGE_WORKER_URL is not set on the backend."}
    headers={"Authorization":f"Bearer {WORKER_TOKEN}"} if WORKER_TOKEN else {}
    try:
        r=requests.get(WORKER_URL+"/health",headers=headers,timeout=20); r.raise_for_status()
        return {"success":True,"configured":True,"provider":"existing_gpu_worker","worker":r.json()}
    except requests.RequestException as exc:
        return {"success":False,"configured":True,"error":str(exc)}

def worker_call(payload):
    if not WORKER_URL: raise RuntimeError("GPU worker is not configured. Set IMAGE_WORKER_URL on the backend service.")
    headers={"Content-Type":"application/json"}
    if WORKER_TOKEN: headers["Authorization"]=f"Bearer {WORKER_TOKEN}"
    try:
        r=requests.post(WORKER_URL+"/generate",json=payload,headers=headers,timeout=TIMEOUT)
        r.raise_for_status(); body=r.json()
        if not body.get("success") or not body.get("image_base64"):
            raise RuntimeError(body.get("error","GPU worker returned no image."))
        return base64.b64decode(body["image_base64"])
    except requests.RequestException as exc:
        raise RuntimeError(f"GPU worker request failed: {exc}") from exc

def save_png(data, filename):
    path=OUTPUT_DIR/filename; path.write_bytes(data)
    return "/outputs/"+filename

def generate_scene_images(job_id: str, data: ImageRequest):
    job=SCENE_JOBS.get(job_id)
    if not job: raise HTTPException(404,"Scene job not found.")
    health=worker_health()
    if not health.get("success"):
        raise HTTPException(503,"GPU worker is unavailable: "+health.get("error","unknown worker error"))
    job["image_generation_status"]="processing"; job["image_generation_error"]=None
    try:
        reference=data.reference_image_base64
        # Use a single-person/character portrait as the img2img continuity anchor.
        if not reference and data.generate_character_reference:
            desc=job.get("character_description") or "main Indian story character, age and appearance suited to the story"
            prompt=("Photorealistic cinematic character continuity portrait, one main character only, "
                    "waist-up, facing camera, neutral gentle expression, plain softly lit background. "
                    f"Character appearance: {desc}. Realistic Indian cinema still, natural skin texture, "
                    "clear face, consistent hairstyle and clothing, no text, no labels, no watermark.")
            raw=worker_call({"prompt":prompt,"negative_prompt":"multiple people, collage, extra faces, text, watermark, distorted face, bad anatomy",
                "width":data.width,"height":data.height,"steps":4,"guidance_scale":0.0,"seed":20261105})
            job["character_reference_url"]=save_png(raw,f"{job_id}_character_reference.png")
            reference=base64.b64encode(raw).decode("utf-8")
        completed=0
        for scene in job["scenes"]:
            scene["image_status"]="processing"
            payload={"prompt":scene["image_prompt"],"negative_prompt":
                "cartoon, illustration, painting, CGI, plastic skin, deformed hands, extra fingers, duplicate people, extra limbs, distorted face, blurry, low quality, text, subtitles, logo, watermark",
                "width":data.width,"height":data.height,"steps":4,"guidance_scale":0.0,
                "seed":-1,"reference_image_base64":reference,"strength":data.strength}
            raw=worker_call(payload)
            scene["image_url"]=save_png(raw,f"{job_id}_scene_{scene['scene_number']}.png")
            scene["image_status"]="completed"; scene["image_error"]=None; completed+=1
        job["image_generation_status"]="completed"; job["images_generated"]=completed
        job["image_provider"]="existing_gpu_worker"
        return {"success":True,"job_id":job_id,"images_generated":completed,
                "character_reference_url":job.get("character_reference_url"),"provider":"existing_gpu_worker","scenes":job["scenes"]}
    except Exception as exc:
        job["image_generation_status"]="failed"; job["image_generation_error"]=str(exc)
        for scene in job["scenes"]:
            if scene.get("image_status")=="processing": scene["image_status"]="failed"; scene["image_error"]=str(exc)
        if isinstance(exc,HTTPException): raise
        raise HTTPException(502,f"Image generation failed: {exc}")

@router.get("/health")
def image_health(): return worker_health()

@router.post("/jobs/{job_id}/generate")
def generate_images(job_id: str, data: ImageRequest): return generate_scene_images(job_id,data)
