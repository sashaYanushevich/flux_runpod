import runpod
from image_generator import generate_image
import base64
from io import BytesIO
from PIL import Image, UnidentifiedImageError

def handler(event):
    job_input = event.get("input", {})
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "No prompt provided."}

    height = job_input.get("height", 512)
    width = job_input.get("width", 512)
    guidance_scale = job_input.get("guidance_scale", 3.5)
    steps = job_input.get("num_inference_steps", 50)
    seed = job_input.get("seed")
    upscale = job_input.get("upscale")  # None / "2x" / "4x"
    strength = job_input.get("strength")
    init_image = None

    image_base64 = job_input.get("image_base64")
    if image_base64:
        try:
            decoded = base64.b64decode(image_base64)
            init_image = Image.open(BytesIO(decoded)).convert("RGB")
        except (ValueError, UnidentifiedImageError) as exc:
            return {"error": f"Invalid base64 image input: {exc}"}

    if strength is not None:
        try:
            strength = float(strength)
            if not 0.0 <= strength <= 1.0:
                raise ValueError("strength must be between 0 and 1")
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid strength value: {exc}"}

    try:
        image = generate_image(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            seed=seed,
            upscale=upscale,
            init_image=init_image,
            strength=strength,
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        return {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
