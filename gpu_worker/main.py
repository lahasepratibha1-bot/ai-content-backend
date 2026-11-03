import base64, io, os
from typing import Optional
import torch
from PIL import Image
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

MODEL_ID = os.getenv("MODEL_ID", "stabilityai/sdxl-turbo")
TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
app = FastAPI(title="Personal AI Video Studio GPU Worker")
text_pipe = None
img_pipe = None

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=6000)
    negative_prompt: str = ""
    width: int = Field(default=576, ge=256, le=1024)
    height: int = Field(default=1024, ge=256, le=1024)
    steps: int = Field(default=1, ge=1, le=8)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=10.0)
    seed: int = -1
    reference_image_base64: Optional[str] = None
    strength: float = Field(default=0.62, ge=0.2, le=0.9)

def auth(value):
    if TOKEN and value != "Bearer " + TOKEN:
        raise HTTPException(401, "Invalid worker token.")

@app.on_event("startup")
def load_models():
    global text_pipe, img_pipe
    kwargs = {"torch_dtype": DTYPE, "use_safetensors": True}
    text_pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, **kwargs)
    img_pipe = AutoPipelineForImage2Image.from_pretrained(MODEL_ID, **kwargs)
    text_pipe.to(DEVICE)
    img_pipe.to(DEVICE)
    if DEVICE == "cuda":
        text_pipe.enable_attention_slicing()
        img_pipe.enable_attention_slicing()

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_ID, "device": DEVICE,
            "cuda_available": torch.cuda.is_available()}

@app.post("/generate")
def generate(data: GenerateRequest, authorization: Optional[str] = Header(default=None)):
    auth(authorization)
    if text_pipe is None or img_pipe is None:
        raise HTTPException(503, "Models are loading.")
    gen = None
    if data.seed >= 0:
        gen = torch.Generator(device=DEVICE).manual_seed(data.seed)
    try:
        if data.reference_image_base64:
            raw = base64.b64decode(data.reference_image_base64)
            init = Image.open(io.BytesIO(raw)).convert("RGB").resize((data.width, data.height))
            result = img_pipe(prompt=data.prompt, negative_prompt=data.negative_prompt or None,
                              image=init, strength=data.strength,
                              num_inference_steps=max(2, data.steps),
                              guidance_scale=data.guidance_scale, generator=gen)
        else:
            result = text_pipe(prompt=data.prompt, negative_prompt=data.negative_prompt or None,
                               width=data.width, height=data.height,
                               num_inference_steps=data.steps,
                               guidance_scale=data.guidance_scale, generator=gen)
        out = io.BytesIO()
        result.images[0].save(out, format="PNG")
        return {"success": True, "image_base64": base64.b64encode(out.getvalue()).decode("utf-8"),
                "width": result.images[0].width, "height": result.images[0].height,
                "model": MODEL_ID}
    except Exception as exc:
        raise HTTPException(500, f"Generation error: {exc}")
