
import re
import uuid
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


def split_into_scenes(script, max_sentences=2):
    sentences = re.split(r'(?<=[.!?।])\s+', script.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    scenes = []
    for i in range(0, len(sentences), max_sentences):
        scenes.append(" ".join(sentences[i:i + max_sentences]))

    return scenes


def make_image_prompt(text, character, style, scene_number):
    character_text = character.strip() or "Characters as described in the scene"

    return (
        f"Create a {style} visual for scene {scene_number}. "
        f"Scene description: {text}. "
        f"Character consistency: {character_text}. "
        "Maintain the same character appearance, clothing, age, "
        "facial features and visual identity across all scenes. "
        "High quality composition, detailed environment, "
        "natural lighting, expressive emotions. "
        "No text, no watermark."
    )


@router.post("/plan")
def create_scene_plan(data: SceneRequest):

    if data.language not in ["hi", "mr", "en"]:
        raise HTTPException(
            status_code=400,
            detail="Supported languages: hi, mr, en"
        )

    if data.video_format not in ["shorts", "landscape"]:
        raise HTTPException(
            status_code=400,
            detail="Format must be shorts or landscape"
        )

    scene_texts = split_into_scenes(data.script)

    if not scene_texts:
        raise HTTPException(
            status_code=400,
            detail="No scenes could be created from script."
        )

    job_id = str(uuid.uuid4())
    scenes = []

    for index, scene_text in enumerate(scene_texts, start=1):

        scene_id = str(uuid.uuid4())

        prompt = make_image_prompt(
            scene_text,
            data.character_description,
            data.visual_style,
            index
        )

        scenes.append({
            "scene_id": scene_id,
            "scene_number": index,
            "scene_text": scene_text,
            "image_prompt": prompt,
            "image_status": "not_generated",
            "image_url": None
        })

    job = {
        "job_id": job_id,
        "status": "planned",
        "language": data.language,
        "video_format": data.video_format,
        "visual_style": data.visual_style,
        "character_description": data.character_description,
        "total_scenes": len(scenes),
        "scenes": scenes,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    SCENE_JOBS[job_id] = job

    return {
        "success": True,
        "message": "Scene plan created. Images are not generated yet.",
        "job": job
    }


@router.get("/jobs")
def get_scene_jobs():

    return {
        "success": True,
        "total_jobs": len(SCENE_JOBS),
        "jobs": list(SCENE_JOBS.values())
    }


@router.get("/jobs/{job_id}")
def get_scene_job(job_id: str):

    job = SCENE_JOBS.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Scene job not found."
        )

    return {
        "success": True,
        "job": job
    }

