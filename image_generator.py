import os
import torch
from PIL import Image
from diffusers import FluxPipeline, StableDiffusionUpscalePipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# 1) FLUX.1-dev
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# 2) SD x4 Upscaler (fp16)
upscaler = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    torch_dtype=torch.float16
)
upscaler.enable_model_cpu_offload()
upscaler.enable_vae_slicing()

def generate_image(
    prompt: str,
    height: int = 512,
    width: int = 512,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 50,
    seed: int | None = None,
    upscale: str | None = None,
):
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    # базовая генерация FLUX
    result = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    img: Image.Image = result.images[0]

    if upscale is None:
        return img

    if upscale == "4x":
        # SD x4 upscaler принимает PIL.Image и промпт
        up = upscaler(prompt=prompt, image=img).images[0]
        return up

    if upscale == "2x":
        # делаем x4, затем даунскейлим до x2 (лучше, чем просто LANCZOS 2x)
        up4 = upscaler(prompt=prompt, image=img).images[0]
        return up4.resize((img.width * 2, img.height * 2), Image.LANCZOS)

    # неизвестный режим — возвращаем базу
    return img
