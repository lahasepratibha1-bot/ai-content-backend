FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV MODEL_ID=stabilityai/sdxl-turbo

COPY gpu_worker/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY gpu_worker/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
