import torch
from diffusers import StableDiffusionPipeline
import time
from torch import autocast

# make sure you're logged in with `huggingface-cli login`
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
)

pipe = pipe.to("cuda")
from torch import autocast
import time

start = time.time()
prompt = "a photograph of an astronaut riding a horse"
with autocast("cuda"):
    images = pipe(prompt, height=512, width=512, batch_size=3).images
    print(images)
end = time.time()
print(end - start)

from torch import autocast
import time

start = time.time()
prompt = "a photograph of an astronaut riding a horse"
with autocast("cuda"):
    images = pipe(prompt, height=512, width=512, batch_size=3).images
    print(images)
end = time.time()
print(end - start)
