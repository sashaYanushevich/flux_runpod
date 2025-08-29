# уже с установленными torch/torchvision под CUDA 12.1
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt constraints.txt /app/

# Сначала обновим pip, потом ставим библиотеки под жёсткие constraints
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Скачаем веса для апскейла, чтобы не тянуть на старте
RUN mkdir -p /app/weights \
 && curl -L -o /app/weights/RealESRGAN_x4plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0/RealESRGAN_x4plus.pth \
 && curl -L -o /app/weights/RealESRGAN_x2plus.pth \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0/RealESRGAN_x2plus.pth

COPY . /app/
CMD ["python", "-u", "rp_handler.py"]
