
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator


router = APIRouter(
    prefix="/api/scenes",
    tags=["AI Scene Engine"]
)

# In-memory scene job storage
SCENE_JOBS: Dict[str, Dict[str, Any]] = {}


class SceneRequest(BaseModel):
    script: str = Field(min_length=1, max_length=50000)
    language: str = "hi"
    video_format: str = "shorts"
    character_description: str = ""
    visual_style: str = "cinematic realistic"
    target_duration: int = Field(default=60, ge=5, le=3600)

    @field_validator("script")
    @classmethod
    def validate_script(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Script cannot be empty.")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value):
        value = value.lower().strip()
        if value not in ["hi", "mr", "en"]:
            raise ValueError("Supported languages: hi, mr, en")
        return value

    @field_validator("video_format")
    @classmethod
    def validate_video_format(cls, value):
        value = value.lower().strip()
        if value not in ["shorts", "landscape"]:
            raise ValueError("Format must be shorts or landscape")
        return value


def split_sentences(script: str) -> List[str]:
    """Split Hindi, Marathi and English scripts into scenes."""
    parts = re.split(r"(?<=[.!?।॥])\s+|\n+", script.strip())
    return [part.strip() for part in parts if part.strip()]


def detect_emotion(text: str) -> str:
    s = text.lower()

    if any(x in s for x in [
        "खुशी के आंसू", "खुशी के आँसू", "खुशी",
        "मुस्कुर", "हैप्पी", "happy", "smile"
    ]):
        return "happy"

    if any(x in s for x in [
        "रोई", "रोया", "दुख", "उदास", "आंसू",
        "आँसू", "sad", "cry", "रडली", "रडला"
    ]):
        return "sad"

    if any(x in s for x in [
        "गुस्सा", "चिल्ला", "नाराज", "रागाव",
        "रागाने", "angry"
    ]):
        return "angry"

    if any(x in s for x in [
        "प्यार", "मां", "माँ", "आई", "बेटी",
        "गले लगा", "प्रेम", "love", "mother",
        "daughter"
    ]):
        return "love"

    if any(x in s for x in [
        "डर", "घबर", "डरी", "डरा", "भीती", "scared"
    ]):
        return "fear"

    return "natural"


def detect_action(text: str) -> str:
    s = text.lower()

    rules = [
        ("waking_up", [
            "उठी", "उठा", "जागी", "जागा", "उठती"
        ]),
        ("hugging", [
            "गले लगा", "लिपट", "आलिंगन", "hug"
        ]),
        ("cooking", [
            "खाना बनाती", "खाना बनाता", "बनाती",
            "बनाता", "स्वयंपाक", "cooking"
        ]),
        ("crying", [
            "रोई", "रोया", "रोने", "रडली",
            "रडला", "crying"
        ]),
        ("smiling", [
            "मुस्कुर", "हंसी", "हँसी", "हसली",
            "हसला", "smile"
        ]),
        ("walking", [
            "चलती", "चला", "गई", "गया",
            "चालली", "गेली", "walking"
        ]),
        ("speaking", [
            "बोली", "बोला", "कहा", "पूछा",
            "म्हणाली", "म्हणाला", "सांगितले"
        ]),
        ("working", [
            "काम करती", "काम करता",
            "काम करते", "काम करत", "working"
        ]),
    ]

    for action, words in rules:
        if any(word in s for word in words):
            return action

    return "natural_action"


def create_scene_plan(data: SceneRequest):
    """Create a structured scene plan from a story script."""

    texts = split_sentences(data.script)

    if not texts:
        raise HTTPException(
            status_code=400,
            detail="No scenes could be created from the script."
        )

    job_id = str(uuid.uuid4())
    scenes = []

    character = (
        data.character_description.strip()
        or "Consistent realistic Indian characters with natural appearance"
    )

    for index, text in enumerate(texts, start=1):

        action = detect_action(text)
        emotion = detect_emotion(text)

        ratio = (
            "9:16 vertical portrait"
            if data.video_format == "shorts"
            else "16:9 horizontal landscape"
        )

        prompt = (
            f"Photorealistic live-action Indian cinematic scene. "
            f"{ratio} format. Scene {index}. "
            f"Story moment: {text}. "
            f"Characters: {character}. "
            f"Visible action: {action}. "
            f"Facial emotion: {emotion}. "
            f"Visual style: {data.visual_style}. "
            "Maintain identical character face, age, hairstyle, "
            "clothing and appearance across every scene. "
            "Natural human anatomy, realistic skin texture, "
            "natural expressions, cinematic lighting, "
            "detailed Indian environment. "
            "No text, no subtitles, no logo, no watermark."
        )

        motion_prompt = (
            f"Animate the scene naturally. Action: {action}. "
            f"Emotion: {emotion}. Story context: {text}. "
            "Show realistic eye movement, breathing, subtle "
            "hand gestures, natural body movement and "
            "controlled cinematic camera motion. "
            "Maintain stable character identity and smooth motion."
        )

        duration = max(
            3,
            min(10, round(len(text.split()) / 2.5))
        )

        scene = {
            "scene_id": str(uuid.uuid4()),
            "scene_number": index,
            "scene_text": text,
            "duration_seconds": duration,
            "characters": character,
            "action": action,
            "emotion": emotion,
            "location": "To be identified by AI",
            "camera_direction": "Cinematic medium shot with subtle movement",
            "dialogue_or_voiceover": text,
            "image_prompt": prompt,
            "motion_prompt": motion_prompt,
            "image_status": "not_generated",
            "video_status": "not_generated",
            "image_url": None,
            "video_url": None
        }

        scenes.append(scene)

    job = {
        "job_id": job_id,
        "status": "planned",
        "planner_type": "rule_based_v2_2",
        "language": data.language,
        "video_format": data.video_format,
        "visual_style": data.visual_style,
        "target_duration": data.target_duration,
        "character_description": character,
        "total_scenes": len(scenes),
        "scenes": scenes,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    SCENE_JOBS[job_id] = job

    return {
        "success": True,
        "message": "Scene plan created successfully",
        "job": job
    }


@router.post("/plan")
def plan_scenes(data: SceneRequest):
    return create_scene_plan(data)


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
            detail="Scene job not found"
        )

    return {
        "success": True,
        "job": job
    }
