import base64
import binascii
import os
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
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=768, ge=256, le=1024)
    strength: float = Field(default=0.62, ge=0.2, le=0.9)
    # The current Colab worker is text-to-image only. Keep reference fields
    # in the API contract, but do not silently pretend to use them.
    generate_character_reference: bool = False
    reference_image_base64: Optional[str] = None


def worker_call(payload: dict) -> bytes:
    if not WORKER_URL:
        raise HTTPException(503, "GPU worker is not configured: IMAGE_WORKER_URL is empty.")

    headers = {"Content-Type": "application/json"}
    if WORKER_TOKEN:
        headers["Authorization"] = f"Bearer {WORKER_TOKEN}"

    try:
        response = requests.post(
            f"{WORKER_URL}/generate",
            json=payload,
            headers=headers,
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise HTTPException(502, f"GPU worker connection failed: {exc}") from exc

    if not response.ok:
        detail = response.text[:1500]
        raise HTTPException(
            502,
            f"GPU worker returned HTTP {response.status_code}: {detail}",
        )

    try:
        result = response.json()
    except ValueError as exc:
        raise HTTPException(502, "GPU worker returned non-JSON response.") from exc

    if not isinstance(result, dict):
        raise HTTPException(502, "GPU worker response must be a JSON object.")

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


def save_png(data: bytes, filename: str) -> str:
    (OUTPUT_DIR / filename).write_bytes(data)
    return "/outputs/" + filename


@router.get("/health")
def image_health():
    if not WORKER_URL:
        return {"success": True, "configured": False, "status": "waiting_for_gpu_worker"}

    headers = {"Authorization": f"Bearer {WORKER_TOKEN}"} if WORKER_TOKEN else {}
