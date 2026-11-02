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
    target_duration: int = Field(default=60, ge=5, le=3600)


def split_sentences(script):
    parts = re.split(r'(?<=[.!?।])\s+|\n+', script.strip())
    return [part.strip() for part in parts if part.strip()]


def detect_emotion(text):
    value = text.lower()

    patterns = {
        "sad": ["रोई", "रोया", "दुख", "आंसू", "अकेली", "अकेला",
                "sad", "cry", "tears", "दुःख"],
        "happy": ["खुश", "हंसी", "मुस्कुर", "आनंद", "happy", "smile"],
        "angry": ["गुस्सा", "चिल्ला", "नाराज", "angry", "shout"],
        "love": ["प्यार", "मां", "माँ", "बेटी", "गले लगा",
                 "love", "hug", "mother", "daughter"],
        "surprised": ["हैरान", "अचानक", "चौंक", "surprised"]
    }

    for emotion, words in patterns.items():
        if any(word in value for word in words):
            return emotion

    return "natural"


def detect_action(text):
    value = text.lower()

    actions = [
        ("waking_up", ["उठी", "उठा", "जागी", "जागा", "wake up"]),
        ("walking", ["चलने", "चलती", "चला", "गई", "गया", "walking"]),
        ("speaking", ["बोली", "बोला", "कहा", "पूछा", "कहती", "speaking"]),
        ("crying", ["रोई", "रोया", "रोने", "आंसू", "crying"]),
        ("hugging", ["गले लगा", "लिपट", "hug", "embrace"]),
        ("smiling", ["मुस्कुर", "हंसी", "हंसा", "smile"]),
        ("working", ["काम", "खाना बनाया", "बनाती", "working"])
    ]

    for action, words in actions:
        if any(word in value for word in words):
            return action

    return "natural_action"


def make_scene_prompt(
    scene_text,
    character,
    style,
    scene_number,
    emotion,
    action,
    video_format
):
    character_info = (
        character.strip()
        if character.strip()
        else "Use the same characters consistently throughout the story."
    )

    ratio = "9:16 vertical" if video_format == "shorts" else "16:9 landscape"

    return (
        f"{style} cinematic scene, {ratio}. "
        f"Scene {scene_number}. "
        f"Story action: {scene_text}. "
        f"Character reference: {character_info}. "
        f"Action: {action}. Emotion: {emotion}. "
        "Preserve identical faces, age, hairstyle, clothing and body proportions "
        "across every scene. Natural expressions, realistic anatomy, "
        "detailed environment, cinematic lighting. "
        "No captions, no text, no watermark."
    )


def make_motion_prompt(action, emotion, scene_text):
    return (
        f"Animate this scene naturally. "
        f"Main action: {action}. "
        f"Emotion: {emotion}. "
        f"Scene context: {scene_text}. "
        "Use realistic body movement, natural facial expressions, "
        "stable character identity and smooth camera movement. "
        "Avoid face morphing, sudden zooms and character duplication."
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

    sentences = split_sentences(data.script)

    if not sentences:
        raise HTTPException(
            status_code=400,
            detail="No scenes could be created from script."
        )

    # Initially, each sentence becomes one scene.
    # Later this can be replaced with an AI semantic scene planner.
    scenes = []
    job_id = str(uuid.uuid4())

    for index, sentence in enumerate(sentences, start=1):
        emotion = detect_emotion(sentence)
        action = detect_action(sentence)

        image_prompt = make_scene_prompt(
            sentence,
            data.character_description,
            data.visual_style,
            index,
            emotion,
            action,
            data.video_format
        )

        motion_prompt = make_motion_prompt(
            action,
            emotion,
            sentence
        )

        scenes.append({
            "scene_id": str(uuid.uuid4()),
            "scene_number": index,
            "scene_text": sentence,
            "duration_seconds": max(
                3,
                min(10, round(len(sentence.split()) / 2.5))
            ),
            "characters": data.character_description,
            "action": action,
            "emotion": emotion,
            "location": "To be identified by AI",
            "camera_direction": "Cinematic medium shot",
            "dialogue_or_voiceover": sentence,
            "image_prompt": image_prompt,
            "motion_prompt": motion_prompt,
            "image_status": "not_generated",
            "video_status": "not_generated",
            "image_url": None,
            "video_url": None
        })

    job = {
        "job_id": job_id,
        "status": "planned",
        "planner_type": "rule_based_v1",
        "language": data.language,
        "video_format": data.video_format,
        "visual_style": data.visual_style,
        "target_duration": data.target_duration,
        "character_description": data.character_description,
        "total_scenes": len(scenes),
        "scenes": scenes,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    SCENE_JOBS[job_id] = job

    return {
        "success": True,
        "message": "Structured scene plan created. AI images and videos are not generated yet.",
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
