import os
import torch
from diffusers import FluxPipeline
from RealESRGAN import RealESRGAN
from PIL import Image

# Устройство для вычислений: используем GPU, если доступен
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем пайплайн модели FLUX.1 [dev] через Diffusers
# (необходимо принять лицензию и иметь доступ к репозиторию модели)
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16,               # используем 16-битный формат (bfloat16) для экономии памяти
    token=huggingface_token          # токен доступа для HuggingFace (если модель приватная)
)
pipe.enable_model_cpu_offload()  # включаем выгрузку частей модели в CPU для экономии GPU памяти:contentReference[oaicite:4]{index=4}

# Кэш моделей ESRGAN, чтобы не загружать повторно при многократном использовании
esrgan_models = {}

def generate_image(prompt: str, height: int = 512, width: int = 512, 
                   guidance_scale: float = 3.5, num_inference_steps: int = 50, 
                   seed: int = None, upscale: str = None):
    """
    Генерирует изображение по текстовому описанию с помощью Flux. 
    Если указан upscale ('2x' или '4x'), выполняется дополнительное улучшение разрешения через ESRGAN.
    """
    # Настраиваем генератор случайности для воспроизводимости (если задан seed)
    generator = None
    if seed is not None:
        generator = torch.Generator(device='cpu').manual_seed(seed)

    # Шаг 1: Генерация изображения моделью FLUX.1 [dev]
    result = pipe(
        prompt, 
        height=height, width=width, 
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps, 
        generator=generator
    )
    image = result.images[0]  # получаем сгенерированный PIL.Image

    # Шаг 2: Опциональное увеличение разрешения с помощью Real-ESRGAN
    if upscale in ['2x', '4x']:
        scale_factor = 4 if upscale == '4x' else 2

        # Загружаем модель ESRGAN нужного масштаба (если не загружена ранее)
        if scale_factor not in esrgan_models:
            esrgan_model = RealESRGAN(device, scale=scale_factor)
            weight_name = f"RealESRGAN_x{scale_factor}.pth"
            esrgan_model.load_weights(f"weights/{weight_name}", download=True)
            esrgan_models[scale_factor] = esrgan_model
        else:
            esrgan_model = esrgan_models[scale_factor]

        # Убеждаемся, что изображение в формате RGB (требуется для ESRGAN)
        image = image.convert("RGB")
        # Прогоняем через ESRGAN для апскейла
        sr_image = esrgan_model.predict(image)
        return sr_image  # возвращаем улучшенное изображение

    # Если upscale не запрошен, возвращаем исходное сгенерированное изображение
    return image
