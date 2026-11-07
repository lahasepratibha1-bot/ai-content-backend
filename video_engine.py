"""Backend client for the upgraded GPU worker's LTX text-to-video endpoint."""
import base64
import os
from pathlib import Path
from typing import Dict, Any
import requests
from fastapi import HTTPException

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
WORKER_URL = os.getenv("IMAGE_WORKER_URL", "").rstrip("/")
WORKER_TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "")
TIMEOUT = int(os.getenv("VIDEO_WORKER_TIMEOUT", "900"))


def _headers():
    h = {"Content-Type": "application/json"}
    if WORKER_TOKEN:
        h["Authorization"] = f"Bearer {WORKER_TOKEN}"
    return h


def video_worker_health():
    if not WORKER_URL:
        return {"success": False, "configured": False,
                "error": "IMAGE_WORKER_URL is not set. Configure the upgraded GPU worker URL."}
    try:
        r = requests.get(WORKER_URL + "/health", headers=_headers(), timeout=30)
        r.raise_for_status()
        body = r.json()
        supported = bool(body.get("video_generation_supported"))
        return {"success": supported, "configured": True, "worker": body,
                "error": body.get("reason") if not supported else None}
    except requests.RequestException as exc:
        return {"success": False, "configured": True, "error": f"GPU worker health check failed: {exc}"}


def _frame_count(seconds: float, fps: int = 24) -> int:
    # LTX-Video temporal length is conventionally 8n+1; keep clips short for VRAM.
    target = max(2.0, min(6.0, seconds))
    choices = [(49, 49 / fps), (73, 73 / fps), (97, 97 / fps), (121, 121 / fps), (145, 145 / fps)]
    return min(choices, key=lambda x: abs(x[1] - target))[0]


def _prompt(scene: Dict[str, Any], video_format: str, visual_style: str) -> str:
    ratio = "vertical 9:16 portrait video" if video_format == "shorts" else "horizontal 16:9 landscape video"
    character = scene.get("characters") or "consistent Indian characters, natural age-appropriate appearance"
    action = scene.get("action") or "natural everyday movement"
    emotion = scene.get("emotion") or "subtle natural emotion"
    text = scene.get("scene_text") or ""
    camera = scene.get("camera_direction") or "cinematic medium shot"
    return (
        f"Photorealistic live-action Indian cinema, {ratio}. {visual_style}. "
        f"One continuous cinematic shot showing this exact story moment: {text}. "
        f"Character continuity: {character}. Visible physical action: {action}. "
        f"Facial performance: {emotion}; believable eye movement, breathing, natural body language, "
        f"subtle hand and head movement. Camera: {camera}, gentle controlled dolly or handheld movement, "
        "realistic human anatomy, natural skin texture, practical lighting, cinematic depth of field, "
        "film color grading, stable faces, coherent motion from beginning to end. No cuts, no text, "
        "no subtitles, no logos, no watermark."
    )


def generate_scene_videos(job_id: str, scene_job: Dict[str, Any], total_audio_duration: float):
    health = video_worker_health()
    if not health.get("success"):
        raise RuntimeError("Real AI video generation is not ready: " + health.get("error", "unknown GPU worker error"))
    scenes = scene_job.get("scenes") or []
    if not scenes:
        raise RuntimeError("Scene planner returned no scenes.")
    weights = [max(1, len((s.get("scene_text") or "").split())) for s in scenes]
    total_weight = sum(weights)
    generated = []
    for index, (scene, weight) in enumerate(zip(scenes, weights), 1):
        target = total_audio_duration * weight / total_weight
        payload = {
            "prompt": _prompt(scene, scene_job.get("video_format", "shorts"), scene_job.get("visual_style", "cinematic natural lighting")),
            "negative_prompt": "cartoon, anime, illustration, plastic skin, deformed face, extra fingers, extra limbs, flicker, jitter, abrupt camera shake, text, captions, watermark",
            "width": 512 if scene_job.get("video_format") == "shorts" else 896,
            "height": 896 if scene_job.get("video_format") == "shorts" else 512,
            "num_frames": _frame_count(target),
            "fps": 24,
            "steps": int(os.getenv("VIDEO_INFERENCE_STEPS", "30")),
            "guidance_scale": 3.0,
            "seed": int(os.getenv("VIDEO_BASE_SEED", "20261105")) + index,
        }
        try:
            response = requests.post(WORKER_URL + "/generate-video", json=payload,
                                     headers=_headers(), timeout=TIMEOUT)
            response.raise_for_status()
            body = response.json()
            if not body.get("success") or not body.get("video_base64"):
                raise RuntimeError(body.get("error", "Worker returned no video clip."))
            clip_bytes = base64.b64decode(body["video_base64"])
            clip_path = OUTPUT_DIR / f"{job_id}_scene_{index:02d}.mp4"
            clip_path.write_bytes(clip_bytes)
            scene["video_url"] = f"/outputs/{clip_path.name}"
            scene["video_status"] = "completed"
            scene["video_duration_seconds"] = body.get("duration_seconds")
            generated.append(clip_path)
        except Exception as exc:
            scene["video_status"] = "failed"
            scene["video_error"] = str(exc)
            raise RuntimeError(f"Scene {index} video generation failed: {exc}") from exc
    return generated


def compose_cinematic_video(job_id: str, clip_paths, audio_path: Path, video_format: str,
                            resolution: str, audio_duration: float, ffmpeg: str):
    if not clip_paths:
        raise RuntimeError("No moving video clips were generated; final render stopped.")
    list_path = OUTPUT_DIR / f"{job_id}_concat.txt"
    # FFmpeg concat demuxer requires escaped absolute paths.
    lines = []
    total_clip_duration = 0.0
    for p in clip_paths:
        escaped = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        # Read duration from ffprobe-like ffmpeg output.
        import subprocess, re
        probe = subprocess.run([ffmpeg, "-i", str(p), "-hide_banner"], capture_output=True, text=True)
        match = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", probe.stderr or "")
        if match:
            total_clip_duration += int(match.group(1))*3600 + int(match.group(2))*60 + int(match.group(3)) + int(match.group(4))/100
    list_path.write_text("\n".join(lines), encoding="utf-8")
    width, height = (720, 1280) if video_format == "shorts" else (1280, 720)
    if resolution == "1080p":
        width, height = width * 1.5, height * 1.5
    width, height = int(width), int(height)
    extra_hold = max(0.0, audio_duration - total_clip_duration)
    out_path = OUTPUT_DIR / f"{job_id}.mp4"
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps=24,setsar=1,format=yuv420p"
    if extra_hold > 0.15:
        vf += f",tpad=stop_mode=clone:stop_duration={extra_hold:.3f}"
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-i", str(audio_path),
           "-vf", vf, "-map", "0:v:0", "-map", "1:a:0", "-t", f"{audio_duration:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out_path)]
    import subprocess
    run = subprocess.run(cmd, capture_output=True, text=True)
    try:
        list_path.unlink(missing_ok=True)
    except Exception:
        pass
    if run.returncode != 0:
        raise RuntimeError("Cinematic video composition failed: " + (run.stderr[-2000:] if run.stderr else "unknown FFmpeg error"))
    return out_path
