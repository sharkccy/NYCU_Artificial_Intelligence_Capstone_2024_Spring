from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

import random
import string
import re
import os

def generate_bgm(video_no, prompt, model_name = "facebook/musicgen-small"):
    processor = AutoProcessor.from_pretrained(model_name)

    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model = model


    os.mkdir(f"./video_output/{video_no}/bgm")


    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors='pt',
    )

    print("Generate bgm for prompt:" + prompt)
    audio_values = model.generate(**inputs, max_new_tokens=1024)
    # token -> 1503:30s, 256:5s, 512:10s
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(f"./video_output/{video_no}/bgm/bgm.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

