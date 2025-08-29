FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir numpy==1.26.4 \
 && python3 -m pip uninstall -y torch torchvision torchaudio || true \
 && python3 -m pip install --no-cache-dir \
      torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
      --extra-index-url https://download.pytorch.org/whl/cu121 \
 && python3 -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/weights \
 && curl -L -o /app/weights/RealESRGAN_x4plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0/RealESRGAN_x4plus.pth \
 && curl -L -o /app/weights/RealESRGAN_x2plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0/RealESRGAN_x2plus.pth

COPY . /app/
CMD ["python3", "-u", "rp_handler.py"]
