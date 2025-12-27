from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
from torchvision import transforms
import re
import os
import torch
import random
import string
from PIL import Image

def generate_image(video_no, prompt, model="samaritan3dCartoon_v40SDXL.safetensors"):
    pipe = StableDiffusionXLPipeline.from_single_file(f".\model\{model}", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    #pipe.load_lora_weights("..\model\lora", weight_name="xl_more_art-full_v1.safetensors")

    prompts = re.split('\n', prompt)
    pre_positive_prompt = ",3dcartoon,masterpiece"
    pre_negtive_prompt = ",low resolution,nsfw,ugly,bad_anatomy,bad_hands,extra_hands,missing_fingers,broken hand,more than two hands,well proportioned hands,more than two legs,unclear eyes,missing_arms,mutilated,extra limbs,extra legs,cloned face,fused fingers,extra_digit, fewer_digits,extra_digits,jpeg_artifacts,signature,watermark,username,blurry,large_breasts,worst_quality,low_quality,normal_quality,mirror image, Vague"
    os.mkdir(f"./video_output/{video_no}/image")
    for i, prompt in enumerate(prompts, start=1):
        w = 768
        h = 1344
        steps = 20
        prompt = prompt.strip() + pre_positive_prompt
        n_prompt = pre_negtive_prompt
        guidance = 7.5
        print("Generate image for prompt:" + prompt)
        image = pipe(prompt, negative_prompt=n_prompt, height=h,width=w, num_inference_steps=steps, guidance_scale=guidance).images[0]
        new_size = (1080,1920)
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

        resized_image.save(f"./video_output/{video_no}/image/image_{i}.jpg")

