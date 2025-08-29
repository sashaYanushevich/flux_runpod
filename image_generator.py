import os
import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline

# <-- НОВОЕ: Real-ESRGAN от xinntao
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FLUX
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    token=hf_token
)
pipe.enable_model_cpu_offload()

def _get_realesrgan_upsampler(scale: int) -> RealESRGANer:
    """Создаёт upsampler для x2 или x4 с локальными весами."""
    assert scale in (2, 4)
    model = RRDBNet(num_in_ch=3, num_out_ch=3,
                    num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    weight_path = f"/app/weights/RealESRGAN_x{scale}plus.pth"
    return RealESRGANer(
        scale=scale, model_path=weight_path, model=model,
        tile=512, tile_pad=10, pre_pad=0, half=True, device=device
    )

def generate_image(prompt: str, height: int = 512, width: int = 512,
                   guidance_scale: float = 3.5, num_inference_steps: int = 50,
                   seed: int | None = None, upscale: str | None = None):
    generator = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None

    result = pipe(
        prompt,
        height=height, width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    )
    image: Image.Image = result.images[0]

    if upscale in ("2x", "4x"):
        scale = 4 if upscale == "4x" else 2
        upsampler = _get_realesrgan_upsampler(scale)
        # RealESRGANer ожидает ndarray BGR; конвертируем туда-обратно
        bgr = cvt_rgb_pil_to_bgr_np(image)
        sr_bgr, _ = upsampler.enhance(bgr, outscale=scale)
        return Image.fromarray(sr_bgr[:, :, ::-1])  # BGR->RGB

    return image

def cvt_rgb_pil_to_bgr_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))[:, :, ::-1]
