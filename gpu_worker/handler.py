import os
import io
import base64
import torch
import runpod

from PIL import Image
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
)

MODEL_ID = os.getenv(
    "MODEL_ID",
    "stabilityai/sdxl-turbo"
)

DEVICE = "cuda"
DTYPE = torch.float16

text_pipe = None
image_pipe = None


def load_models():
    global text_pipe, image_pipe

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    print("Loading SDXL Turbo model...")

    text_pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        use_safetensors=True
    )

    text_pipe.to(DEVICE)

    image_pipe = AutoPipelineForImage2Image.from_pipe(
        text_pipe
    )

    image_pipe.to(DEVICE)

    print("SDXL Turbo model loaded successfully")


def handler(job):
    data = job.get("input", {})

    prompt = data.get("prompt", "").strip()

    if not prompt:
        return {"success": False, "error": "Prompt is required"}

    negative_prompt = data.get("negative_prompt", "")

    width = int(data.get("width", 512))
    height = int(data.get("height", 512))

    width = max(256, min(width, 1024))
    height = max(256, min(height, 1024))

    width = (width // 8) * 8
    height = (height // 8) * 8

    steps = max(1, min(int(data.get("steps", 1)), 8))
    guidance = float(data.get("guidance_scale", 0.0))

    seed = int(data.get("seed", -1))

    generator = None
    if seed >= 0:
        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(seed)

    try:
        reference = data.get("reference_image_base64")

        if reference:
            image_bytes = base64.b64decode(reference)
            init_image = Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB").resize((width, height))

            strength = float(data.get("strength", 0.62))
            strength = max(0.2, min(strength, 0.9))

            result = image_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=init_image,
                strength=strength,
                num_inference_steps=max(2, steps),
                guidance_scale=guidance,
                generator=generator
            )
        else:
            result = text_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            )

        output = io.BytesIO()
        result.images[0].save(output, format="PNG")

        encoded_image = base64.b64encode(
            output.getvalue()
        ).decode("utf-8")

        return {
            "success": True,
            "image_base64": encoded_image,
            "width": width,
            "height": height,
            "model": MODEL_ID
        }

    except Exception as error:
        print(f"Image generation failed: {error}")
        return {
            "success": False,
            "error": str(error)
        }


if __name__ == "__main__":
    load_models()

    runpod.serverless.start({
        "handler": handler
    })
