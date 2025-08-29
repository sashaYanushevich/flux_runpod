import runpod
from image_generator import generate_image
import base64
from io import BytesIO

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

    try:
        image = generate_image(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            seed=seed,
            upscale=upscale,
        )
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image_base64": img_b64}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
