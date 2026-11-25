"""Render-side API router for the SDXL GPU worker."""
from __future__ import annotations

import base64
import binascii
import os
import re
import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from scene_engine import SCENE_JOBS

router = APIRouter(prefix="/api/images", tags=["AI Image Engine"])
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKER_URL = os.getenv("IMAGE_WORKER_URL", "").strip().rstrip("/")
WORKER_TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "").strip()
try:
    WORKER_TIMEOUT = max(15, int(os.getenv("IMAGE_WORKER_TIMEOUT", "240")))
except ValueError:
    WORKER_TIMEOUT = 240


class ImageRequest(BaseModel):
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=768, ge=256, le=1024)
    steps: int = Field(default=2, ge=1, le=8)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    strength: float = Field(default=0.62, ge=0.2, le=0.9)
    generate_character_reference: bool = False
    reference_image_base64: Optional[str] = None


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if WORKER_TOKEN:
        headers["Authorization"] = f"Bearer {WORKER_TOKEN}"
    return headers


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)[:100] or uuid.uuid4().hex


def _worker_json(path: str, payload: Optional[dict] = None, timeout: int = 20) -> dict:
    if not WORKER_URL:
        raise HTTPException(503, "GPU worker is not configured: IMAGE_WORKER_URL is empty.")
    try:
        if payload is None:
            response = requests.get(f"{WORKER_URL}{path}", headers=_headers(), timeout=timeout)
        else:
            response = requests.post(
                f"{WORKER_URL}{path}", json=payload, headers=_headers(), timeout=WORKER_TIMEOUT
            )
    except requests.RequestException as exc:
        raise HTTPException(502, f"GPU worker connection failed: {exc}") from exc

    if not response.ok:
        detail = response.text[:1500]
        raise HTTPException(502, f"GPU worker returned HTTP {response.status_code}: {detail}")
    try:
        result = response.json()
    except ValueError as exc:
        raise HTTPException(502, "GPU worker returned a non-JSON response.") from exc
    if not isinstance(result, dict):
        raise HTTPException(502, "GPU worker response must be a JSON object.")
    return result


def worker_call(payload: dict) -> bytes:
    result = _worker_json("/generate", payload, timeout=WORKER_TIMEOUT)
    if result.get("success") is False:
        raise HTTPException(502, f"GPU worker generation failed: {result.get('error', 'Unknown error')}")
    encoded = result.get("image_base64")
    if not isinstance(encoded, str) or not encoded.strip():
        raise HTTPException(
            502,
            f"GPU worker response is missing image_base64. Response keys: {list(result.keys())}",
        )
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(502, "GPU worker returned invalid base64 image data.") from exc
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(502, "GPU worker image payload is not a valid PNG file.")
    return image_bytes


def _save_png(data: bytes, filename: str) -> str:
    requested = Path(filename).name
    stem = _safe_stem(Path(requested).stem)
    final_name = f"{stem}.png"
    path = (OUTPUT_DIR / final_name).resolve()
    if OUTPUT_DIR.resolve() not in path.parents:
        raise HTTPException(400, "Invalid output filename.")
    path.write_bytes(data)
    return f"/outputs/{path.name}"


@router.get("/health")
def image_health():
    if not WORKER_URL:
        return {"success": True, "configured": False, "status": "waiting_for_gpu_worker"}
    try:
        worker = _worker_json("/health", timeout=15)
        return {
            "success": True,
            "configured": True,
            "provider": "existing_gpu_worker",
            "worker": worker,
        }
    except HTTPException as exc:
        return {"success": False, "configured": True, "error": exc.detail}


@router.post("/jobs/{job_id}/generate")
def generate_images(job_id: str, data: ImageRequest):
    job = SCENE_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Scene job not found.")
    scenes = job.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise HTTPException(400, "Scene job has no scenes to generate.")
    if not WORKER_URL:
        raise HTTPException(503, "GPU worker is not connected. Configure IMAGE_WORKER_URL.")

    reference_b64 = data.reference_image_base64
    character_reference_url = job.get("character_reference_url")
    job["image_generation_status"] = "processing"
    job["image_generation_error"] = None
    completed = 0

    try:
        if data.generate_character_reference and not reference_b64:
            character_description = str(job.get("character_description", "")).strip()
            if not character_description:
                raise HTTPException(
                    400,
                    "character_description is required to generate a character reference.",
                )
            reference_prompt = (
                "Create a clean full-body character reference portrait, single person only. "
                f"Character appearance: {character_description}. "
                "Neutral natural pose, clearly visible face and clothing, simple uncluttered background, "
                "consistent realistic proportions, no text, no watermark."
            )
            reference_bytes = worker_call({
                "prompt": reference_prompt,
                "negative_prompt": "multiple people, duplicate person, cropped face, text, watermark, distorted anatomy",
                "width": data.width,
                "height": data.height,
                "steps": data.steps,
                "guidance_scale": data.guidance_scale,
            })
            reference_b64 = base64.b64encode(reference_bytes).decode("ascii")
            character_reference_url = _save_png(
                reference_bytes, f"{_safe_stem(job_id)}_character_reference.png"
            )
            job["character_reference_url"] = character_reference_url

        if reference_b64:
            # Verify caller-supplied reference is valid base64 before sending it to the worker.
            try:
                base64.b64decode(reference_b64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(400, "reference_image_base64 is not valid base64.") from exc

        for index, scene in enumerate(scenes, start=1):
            scene["image_status"] = "processing"
            prompt = str(scene.get("image_prompt", "")).strip()
            if not prompt:
                raise HTTPException(400, f"Scene {index} is missing image_prompt.")
            payload = {
                "prompt": prompt,
                "negative_prompt": "different face, different clothes, extra people, duplicate person, bad anatomy, text, watermark, blurry",
                "width": data.width,
                "height": data.height,
                "steps": data.steps,
                "guidance_scale": data.guidance_scale,
                "strength": data.strength,
            }
            if reference_b64:
                payload["reference_image_base64"] = reference_b64
            image_bytes = worker_call(payload)
            scene_number = scene.get("scene_number", index)
            scene["image_url"] = _save_png(
                image_bytes, f"{_safe_stem(job_id)}_scene_{int(scene_number)}.png"
            )
            scene["image_status"] = "completed"
            scene["image_error"] = None
            completed += 1

        job["image_generation_status"] = "completed"
        job["images_generated"] = completed
        return {
            "success": True,
            "job_id": job_id,
            "images_generated": completed,
            "character_reference_url": character_reference_url,
            "scenes": scenes,
        }
    except HTTPException as exc:
        job["image_generation_status"] = "failed"
        job["image_generation_error"] = exc.detail
        for scene in scenes:
            if scene.get("image_status") == "processing":
                scene["image_status"] = "failed"
                scene["image_error"] = exc.detail
        raise
    except Exception as exc:
        job["image_generation_status"] = "failed"
        job["image_generation_error"] = str(exc)
        for scene in scenes:
            if scene.get("image_status") == "processing":
                scene["image_status"] = "failed"
                scene["image_error"] = str(exc)
        raise HTTPException(500, f"Image generation failed: {type(exc).__name__}: {exc}") from exc
