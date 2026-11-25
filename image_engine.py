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
