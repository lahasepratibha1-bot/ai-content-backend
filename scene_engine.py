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
        ("writing", ["writes a letter", "writing a letter", "writes", "writing", "पत्र लिख", "लिहिते"]),
        ("sitting", ["sits", "sitting", "बैठी", "बैठता", "बैठकर", "बसली", "बसतो"]),
        ("walking", ["walks", "walking", "walk", "चलती", "चला", "गई", "गया", "चालली", "गेली"]),
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


def scene_composition(action: str, text: str, ratio: str):
    """Return action-led framing so scenes show the story, not repeated portraits."""
    s = text.lower()
    if action == "walking" or "garden" in s:
        return (
            "Three-quarter or full-body environmental shot, subject visibly walking "
            "through a lush garden, one foot mid-step, arms and hands naturally visible, "
            "garden path and greenery clearly visible behind her. Keep the head, neck, "
            "shoulders and torso in realistic human proportions."
        ), "Three-quarter full-body tracking shot at eye level"
    if action == "writing" or any(x in s for x in ["letter", "table", "writes"]):
        return (
            "Medium-wide waist-up seated composition at a wooden table, both forearms "
            "and hands visible writing on a letter with a pen, tabletop and letter clearly "
            "visible in frame. Natural seated posture, shoulders relaxed, head and neck "
            "proportioned naturally to the torso."
        ), "Medium-wide seated shot, eye-level three-quarter angle"
    if action == "sitting" or any(x in s for x in ["window", "rainy", "sits"]):
        return (
            "Medium-wide waist-up seated composition beside a clearly visible rain-streaked "
            "window, subject looking outside, window frame and raindrops visible in the "
            "background. Natural seated posture with shoulders visible and relaxed, "
            "normal neck length and realistic head-to-body proportions."
        ), "Medium-wide seated profile three-quarter shot"
    return (
        "Medium shot showing head, shoulders and upper torso with natural body proportions, "
        "visible environment relevant to the story, avoid tight head-and-neck close-up."
    ), "Cinematic medium shot at eye level"


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

        composition, camera_direction = scene_composition(action, text, ratio)

        prompt = (
            f"Photorealistic live-action Indian cinematic scene. "
            f"{ratio} format. Scene {index}. "
            f"Story moment: {text}. "
            f"Characters: {character}. "
            f"Visible action: {action}. "
            f"Required scene composition: {composition}. "
            f"Facial emotion: {emotion}. "
            f"Visual style: {data.visual_style}. "
            "Preserve the same character identity, face shape, age, hairstyle and outfit "
            "from the supplied reference image. "
            "Anatomically realistic adult human proportions: average natural neck length "
            "and width, neck connected naturally to visible shoulders, balanced head-to-body "
            "ratio, realistic shoulder width and upper torso. "
            "Use a medium or wider camera framing that shows the stated action and its "
            "environment; do not turn every scene into a centered beauty portrait. "
            "Natural skin texture, natural expression, cinematic lighting, detailed Indian environment. "
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
            "camera_direction": camera_direction,
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
