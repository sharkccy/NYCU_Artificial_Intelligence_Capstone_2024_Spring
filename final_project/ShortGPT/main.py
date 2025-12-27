import os
import string
import random
from utils import Text_to_bgm, Text_to_image,Text_to_speech,Video_editing
from utils import Prompt_generator
import asyncio

# key = "please put your gpt api key here"

video_no = ''.join(random.sample(string.ascii_letters + string.digits, 8))
os.mkdir(f"./video_output/{video_no}")
print(f"Video number is {video_no}")

video_prompt = input("Input video topic prompt:")
script_content, image_prompt, music_prompt = Prompt_generator.generate_prompt(video_no, video_prompt , key)

#Generate speech
asyncio.run(Text_to_speech.generate_speech(video_no, script_content, sub_len=2))
#Generate image
Text_to_image.generate_image(video_no, image_prompt)
#Generate bgm
Text_to_bgm.generate_bgm(video_no, music_prompt)
#Video editting
Video_editing.generate_video(video_no)

#Finish
print("Process succesfully finished.")
print(f"Video has been stored in \"./video_output/{video_no}\"")
