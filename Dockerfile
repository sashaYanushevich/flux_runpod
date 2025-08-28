FROM runpod/base:0.6.2-cuda11.8.0
WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN python3 -m pip uninstall -y torch torchvision torchaudio || true

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir \
      torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN python3 - <<'PY'
import torch, diffusers, huggingface_hub, runpod
print("torch:", torch.__version__)
print("diffusers:", diffusers.__version__)
print("hf_hub:", huggingface_hub.__version__)
print("runpod:", runpod.__version__)
PY

CMD ["python3", "-u", "rp_handler.py"]
