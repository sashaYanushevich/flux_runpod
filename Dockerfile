FROM runpod/base:0.6.2-cuda11.8.0

WORKDIR /app

COPY requirements.txt /app/

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118

RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /app/



CMD ["python3", "-u", "rp_handler.py"]
