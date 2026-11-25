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
    try:
        response = requests.get(f"{WORKER_URL}/health", headers=headers, timeout=15)
        response.raise_for_status()
        return {"success": True, "configured": True, "provider": "existing_gpu_worker", "worker": response.json()}
    except requests.RequestException as exc:
        return {"success": False, "configured": True, "error": str(exc)}
    except ValueError as exc:
        return {"success": False, "configured": True, "error": f"Worker health returned invalid JSON: {exc}"}


@router.post("/jobs/{job_id}/generate")
def generate_images(job_id: str, data: ImageRequest):
    job = SCENE_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Scene job not found.")
    if not WORKER_URL:
        raise HTTPException(503, "GPU worker is not connected. Configure IMAGE_WORKER_URL.")

    # Current Colab deployment loads only AutoPipelineForText2Image.
    # Reject reference-image requests instead of silently ignoring them.
    if data.generate_character_reference or data.reference_image_base64:
        raise HTTPException(
            501,
            "Character-reference / image-to-image generation is not enabled on the current text-to-image GPU worker.",
        )

    job["image_generation_status"] = "processing"
    job["image_generation_error"] = None
    completed = 0

    try:
        for scene in job.get("scenes", []):
            scene["image_status"] = "processing"
            payload = {
                "prompt": scene["image_prompt"],
                "negative_prompt": "different face, different clothes, extra people, duplicate person, bad anatomy, text, watermark, blurry",
                "width": data.width,
                "height": data.height,
                "steps": 2,
                "guidance_scale": 0.0,
            }
            raw = worker_call(payload)
            scene["image_url"] = save_png(raw, f"{job_id}_scene_{scene['scene_number']}.png")
            scene["image_status"] = "completed"
            scene["image_error"] = None
            completed += 1

        job["image_generation_status"] = "completed"
        job["images_generated"] = completed
        return {
            "success": True,
            "job_id": job_id,
            "images_generated": completed,
            "character_reference_url": None,
            "scenes": job.get("scenes", []),
        }
    except HTTPException as exc:
        job["image_generation_status"] = "failed"
        job["image_generation_error"] = exc.detail
        for scene in job.get("scenes", []):
            if scene.get("image_status") == "processing":
                scene["image_status"] = "failed"
                scene["image_error"] = exc.detail
        raise
    except Exception as exc:
        job["image_generation_status"] = "failed"
        job["image_generation_error"] = str(exc)
        for scene in job.get("scenes", []):
            if scene.get("image_status") == "processing":
                scene["image_status"] = "failed"
                scene["image_error"] = str(exc)
        raise HTTPException(500, f"Image generation failed: {exc}") from exc
