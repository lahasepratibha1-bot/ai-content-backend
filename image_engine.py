import base64, os, time, uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scene_engine import SCENE_JOBS

router = APIRouter(prefix="/api/images", tags=["AI Image Engine"])
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Provider A: existing custom GPU worker exposing /generate and /health.
WORKER_URL = os.getenv("IMAGE_WORKER_URL", "").rstrip("/")
WORKER_TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "")
TIMEOUT = int(os.getenv("IMAGE_WORKER_TIMEOUT", "240"))

# Provider B: Runpod Serverless endpoint. It is optional and OFF unless both
# variables are configured. Requests can incur GPU charges.
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "").strip()
RUNPOD_BASE = "https://api.runpod.ai/v2"

class ImageRequest(BaseModel):
    width: int = Field(default=576, ge=256, le=1024)
    height: int = Field(default=1024, ge=256, le=1024)
    strength: float = Field(default=0.62, ge=0.2, le=0.9)
    generate_character_reference: bool = True
    reference_image_base64: Optional[str] = None

def _save_png(data, filename):
    (OUTPUT_DIR / filename).write_bytes(data)
    return "/outputs/" + filename

def _decode_image(value):
    if not isinstance(value, str):
        raise ValueError("Image data was not a string")
    if value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)

def _custom_worker_call(payload):
    headers = {"Content-Type": "application/json"}
    if WORKER_TOKEN:
        headers["Authorization"] = f"Bearer {WORKER_TOKEN}"
    try:
        r = requests.post(WORKER_URL + "/generate", json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        return base64.b64decode(r.json()["image_base64"])
    except requests.RequestException as exc:
        raise HTTPException(502, f"GPU worker request failed: {exc}")
    except (KeyError, ValueError) as exc:
        raise HTTPException(502, f"Invalid custom GPU worker response: {exc}")

def _runpod_call(prompt, negative_prompt, width, height, steps=4):
    """Calls Runpod Serverless /run then polls /status. Disabled unless credentials exist."""
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    base = f"{RUNPOD_BASE}/{RUNPOD_ENDPOINT_ID}"
    # SDXL Turbo Hub worker expects 512x512; it does not accept reference-image conditioning.
    payload = {"input": {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": 512,
        "height": 512,
        "num_inference_steps": min(max(int(steps), 1), 8),
        "guidance_scale": 0.0,
        "num_images": 1
    }}
    try:
        response = requests.post(base + "/run", json=payload, headers=headers, timeout=45)
        response.raise_for_status()
        job = response.json()
        request_id = job.get("id")
        if not request_id:
            raise ValueError("Runpod did not return a job id")
        deadline = time.time() + TIMEOUT
        while time.time() < deadline:
            status_response = requests.get(base + f"/status/{request_id}", headers=headers, timeout=30)
            status_response.raise_for_status()
            status_data = status_response.json()
            status = status_data.get("status")
            if status == "COMPLETED":
                output = status_data.get("output") or {}
                images = output.get("images") or []
                if not images:
                    raise ValueError("Runpod completed but returned no images")
                return _decode_image(images[0].get("image"))
            if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                raise RuntimeError(f"Runpod job {status}: {status_data.get('error', 'no error detail')}")
            time.sleep(2)
        raise TimeoutError("Runpod image job exceeded IMAGE_WORKER_TIMEOUT")
    except requests.RequestException as exc:
        raise HTTPException(502, f"Runpod request failed: {exc}")
    except (KeyError, ValueError, RuntimeError, TimeoutError) as exc:
        raise HTTPException(502, f"Runpod image generation failed: {exc}")

def worker_call(payload):
    if WORKER_URL:
        return _custom_worker_call(payload)
    if RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY:
        return _runpod_call(
            payload["prompt"], payload.get("negative_prompt", ""),
            payload.get("width", 512), payload.get("height", 512),
            payload.get("steps", 4)
        )
    raise HTTPException(503, "No image provider configured. GPU generation is intentionally disabled until you configure a worker.")

@router.get("/health")
def image_health():
    if WORKER_URL:
        try:
            headers = {"Authorization": f"Bearer {WORKER_TOKEN}"} if WORKER_TOKEN else {}
            r = requests.get(WORKER_URL + "/health", headers=headers, timeout=15)
            r.raise_for_status()
            return {"success": True, "configured": True, "provider": "custom_worker", "worker": r.json()}
        except requests.RequestException as exc:
            return {"success": False, "configured": True, "provider": "custom_worker", "error": str(exc)}
    if RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY:
        return {"success": True, "configured": True, "provider": "runpod_serverless",
                "endpoint_id_present": True, "note": "Calls may incur GPU charges."}
    return {"success": True, "configured": False, "status": "waiting_for_gpu_worker",
            "note": "No GPU calls will be made; no GPU charges."}

@router.post("/jobs/{job_id}/generate")
def generate_images(job_id: str, data: ImageRequest):
    job = SCENE_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Scene job not found.")
    if not WORKER_URL and not (RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY):
        raise HTTPException(503, "GPU worker is not connected. Image generation remains disabled.")
    job["image_generation_status"] = "processing"
    job["image_generation_error"] = None
    try:
        reference = data.reference_image_base64
        # Character-reference sheet works only with the custom worker that supports image-to-image.
        # The Runpod Hub SDXL Turbo worker does not document reference-image input.
        if WORKER_URL and not reference and data.generate_character_reference:
            desc = job.get("character_description") or "the main story characters"
            prompt = ("Character reference sheet, full body, front view, neutral simple background. "
                      + desc + ". Show only the described characters, clearly separated. "
                      "Keep faces, age, hairstyle and clothing visible. " +
                      job.get("visual_style", "cinematic realistic") +
                      " style, no text, no labels, no watermark.")
            raw = worker_call({"prompt": prompt, "negative_prompt": "extra people, crowd, duplicate person, text, watermark",
                               "width": data.width, "height": data.height, "steps": 1,
                               "guidance_scale": 0.0, "seed": -1})
            job["character_reference_url"] = _save_png(raw, f"{job_id}_character_reference.png")
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
            scene["image_url"] = _save_png(raw, f"{job_id}_scene_{scene['scene_number']}.png")
            scene["image_status"] = "completed"
            scene["image_error"] = None
            completed += 1
        job["image_generation_status"] = "completed"
        job["images_generated"] = completed
        return {"success": True, "job_id": job_id, "images_generated": completed,
                "character_reference_url": job.get("character_reference_url"),
                "provider": "custom_worker" if WORKER_URL else "runpod_serverless",
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
