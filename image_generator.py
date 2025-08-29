import os
import torch
from PIL import Image
from diffusers import FluxPipeline, StableDiffusionUpscalePipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

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
    upscale: str | None = None,  # None | "2x" | "4x"
):
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    res = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    img: Image.Image = res.images[0]

    if upscale == "4x":
        return upscaler(prompt=prompt, image=img).images[0]
    if upscale == "2x":
        up4 = upscaler(prompt=prompt, image=img).images[0]
        return up4.resize((img.width * 2, img.height * 2), Image.LANCZOS)

    return img
