import keras_cv
import time
from tensorflow.keras import mixed_precision


mixed_precision.set_global_policy("mixed_float16")
supermodel = keras_cv.models.StableDiffusion(jit_compile=True)

start = time.time()
prompt = "a photograph of an astronaut riding a horse"
images = supermodel.text_to_image(prompt, batch_size=3)
end = time.time()
print(end - start)

start = time.time()
prompt = "a photograph of an astronaut riding a horse"
images = supermodel.text_to_image(prompt, batch_size=3)
end = time.time()
print(end - start)
