PERSONAL AI VIDEO STUDIO — FINAL INTEGRATED IMAGE ENGINE PACKAGE

GITHUB ROOT: replace main.py, scene_engine.py, image_engine.py, requirements.txt.
Keep gpu_worker folder and upload gpu_worker/main.py + gpu_worker/requirements.txt.

RENDER BACKEND SETTINGS:
Build: pip install -r requirements.txt
Start: uvicorn main:app --host 0.0.0.0 --port $PORT

GPU WORKER:
Deploy gpu_worker as a separate Python service on a CUDA GPU host.
Install gpu_worker/requirements.txt.
Start command: uvicorn main:app --host 0.0.0.0 --port 8000
Worker env: MODEL_ID=stabilityai/sdxl-turbo
Worker env: IMAGE_WORKER_TOKEN=<create a long random secret>
Render env: IMAGE_WORKER_URL=https://<GPU-worker-public-domain>
Render env: IMAGE_WORKER_TOKEN=<same secret>
Render env: IMAGE_WORKER_TIMEOUT=240
Never commit the token to GitHub.

API:
1. POST /api/scenes/plan
2. POST /api/images/jobs/{job_id}/generate
Body: {"width":576,"height":1024,"strength":0.62,"generate_character_reference":true}

LIMITATIONS:
This image engine creates still images, not moving video. Character reference conditioning may help continuity but does not guarantee identical faces/clothes or exact actions. Jobs are in memory and Render Free output storage is ephemeral. The image endpoint is synchronous and long scripts may time out. GPU service needs a CUDA GPU and downloads SDXL Turbo weights on first start. This is a development integration, not production-grade durable processing/storage.
