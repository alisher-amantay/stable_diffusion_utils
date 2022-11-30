import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

PATH = 'generations/'
n_images = 1
prompt = "a detailed wallpaper of a basement covered in body parts, intestines and blood in horror catholic gothic style, hyperrealistic, extreme details, trending on artstation"
model_id = '/raid/alisher_amantay/.cache/huggingface/diffusers/models--CompVis--stable-diffusion-v1-4/snapshots/2881c082ee0dc70d9eeb645f1b150040a4b62767/'
# 'stable-diffusion-v1-4'
device = 'cuda:0'

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=lms, use_auth_token=True)
pipe = pipe.to(device)

# run pipeline in inference (sample random noise and denoise)
with autocast('cuda'):
    images = pipe([prompt]*n_images, num_inference_steps=50, height=512, width=960,
                  eta=0.0, guidance_scale=7.0)["sample"]

# save images
for idx, image in enumerate(images):
    image.save(PATH + f'{prompt}_{idx}.png')