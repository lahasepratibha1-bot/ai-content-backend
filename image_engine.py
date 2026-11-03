import base64, os, uuid
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
TIMEOUT = int(os.getenv("IMAGE_WORKER_TIMEOUT", "240"))

class ImageRequest(BaseModel):
    width: int = Field(default=576, ge=256, le=1024)
    height: int = Field(default=1024, ge=256, le=1024)
    strength: float = Field(default=0.62, ge=0.2, le=0.9)
    generate_character_reference: bool = True
    reference_image_base64: Optional[str] = None

def worker_call(payload):
    if not WORKER_URL:
        raise HTTPException(503, "Set IMAGE_WORKER_URL in Render environment variables.")
    headers = {"Content-Type": "application/json"}
    if WORKER_TOKEN:
        headers["Authorization"] = f"Bearer {WORKER_TOKEN}"
    try:
        r = requests.post(WORKER_URL + "/generate", json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        result = r.json()
        return base64.b64decode(result["image_base64"])
    except requests.RequestException as exc:
        raise HTTPException(502, f"GPU worker request failed: {exc}")
    except (KeyError, ValueError) as exc:
        raise HTTPException(502, f"Invalid GPU worker response: {exc}")

def save_png(data, filename):
    (OUTPUT_DIR / filename).write_bytes(data)
    return "/outputs/" + filename

@router.get("/health")
def image_health():
    if not WORKER_URL:
        return {"success": True, "configured": False, "status": "waiting_for_gpu_worker"}
    try:
        headers = {"Authorization": f"Bearer {WORKER_TOKEN}"} if WORKER_TOKEN else {}
        r = requests.get(WORKER_URL + "/health", headers=headers, timeout=15)
        r.raise_for_status()
        return {"success": True, "configured": True, "worker": r.json()}
    except requests.RequestException as exc:
        return {"success": False, "configured": True, "error": str(exc)}

@router.post("/jobs/{job_id}/generate")
def generate_images(job_id: str, data: ImageRequest):
    job = SCENE_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Scene job not found.")
    if not WORKER_URL:
        raise HTTPException(503, "GPU worker is not connected. Configure IMAGE_WORKER_URL.")
    job["image_generation_status"] = "processing"
    job["image_generation_error"] = None
    try:
        reference = data.reference_image_base64
        if not reference and data.generate_character_reference:
            desc = job.get("character_description") or "the main story characters"
            prompt = ("Character reference sheet, full body, front view, neutral simple background. "
                      + desc + ". Show only the described characters, clearly separated. "
                      "Keep faces, age, hairstyle and clothing visible. " +
                      job.get("visual_style", "cinematic realistic") +
                      " style, no text, no labels, no watermark.")
            raw = worker_call({"prompt": prompt, "negative_prompt": "extra people, crowd, duplicate person, text, watermark",
                               "width": data.width, "height": data.height, "steps": 1,
                               "guidance_scale": 0.0, "seed": -1})
            job["character_reference_url"] = save_png(raw, f"{job_id}_character_reference.png")
            reference = base64.b64encode(raw).decode("utf-8")

        completed = 0
        for scene in job["scenes"]:
            scene["image_status"] = "processing"
            payload = {"prompt": scene["image_prompt"],
                       "negative_prompt": "different face, different clothes, extra people, duplicate person, bad anatomy, text, watermark, blurry",
                       "width": data.width, "height": data.height, "steps": 4,
                       "guidance_scale": 0.0, "seed": -1,
                       "reference_image_base64": reference, "strength": data.strength}
            raw = worker_call(payload)
            scene["image_url"] = save_png(raw, f"{job_id}_scene_{scene['scene_number']}.png")
            scene["image_status"] = "completed"
            scene["image_error"] = None
            completed += 1
        job["image_generation_status"] = "completed"
        job["images_generated"] = completed
        return {"success": True, "job_id": job_id, "images_generated": completed,
                "character_reference_url": job.get("character_reference_url"),
                "scenes": job["scenes"]}
    except Exception as exc:
        job["image_generation_status"] = "failed"
        job["image_generation_error"] = str(exc)
        for scene in job["scenes"]:
            if scene.get("image_status") == "processing":
                scene["image_status"] = "failed"
                scene["image_error"] = str(exc)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(500, f"Image generation failed: {exc}")
