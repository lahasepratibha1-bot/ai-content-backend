FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV MODEL_ID=stabilityai/sdxl-turbo
ENV VIDEO_MODEL_ID=Lightricks/LTX-Video
ENV MIN_VIDEO_VRAM_GB=10
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY main.py /app/main.py
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
