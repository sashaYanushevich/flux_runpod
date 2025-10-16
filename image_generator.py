import os
import torch
from PIL import Image
from diffusers import FluxPipeline, FluxImg2ImgPipeline, StableDiffusionUpscalePipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_flux_dtype() -> torch.dtype:
    if device.type != "cuda":
        return torch.float32

    try:
        major, _minor = torch.cuda.get_device_capability()
    except RuntimeError:
        return torch.float16

    if major >= 8:
        return torch.bfloat16

    return torch.float16


def _select_upscaler_dtype(flux_dtype: torch.dtype) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if flux_dtype == torch.float32:
        return torch.float32
    return torch.float16


HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
FLUX_DTYPE = _select_flux_dtype()
UPSCALE_DTYPE = _select_upscaler_dtype(FLUX_DTYPE)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=FLUX_DTYPE,
    token=HF_TOKEN
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=FLUX_DTYPE,
    token=HF_TOKEN
)
img2img_pipe.enable_model_cpu_offload()
img2img_pipe.enable_vae_slicing()

upscaler = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler",
    torch_dtype=UPSCALE_DTYPE
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
    init_image: Image.Image | None = None,
    strength: float | None = None,
):
    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    if init_image is not None:
        img2img_kwargs = {
            "prompt": prompt,
            "image": init_image,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }
        if strength is not None:
            img2img_kwargs["strength"] = strength
        res = img2img_pipe(**img2img_kwargs)
    else:
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
