#!/usr/bin/env python3
"""
Streaming TTS example with subtitles.

This example is similar to the example basic_audio_streaming.py, but it shows
WordBoundary events to create subtitles using SubMaker.
"""

import asyncio
import edge_tts
import re
import os
import random
import string




error_switcher = {
        "en-US-RogerNeural": 9000000,
        "en-US-AnaNeural": 9250000,
    }


async def generate_speech(video_no, input_text, sub_len = 6, voice = "en-US-RogerNeural") -> None:

    """Main function"""

    communicate = edge_tts.Communicate(input_text, voice)
    submaker = edge_tts.SubMaker()
    os.mkdir(f"./video_output/{video_no}/speech")
    output_file = f"./video_output/{video_no}/speech/speech.mp3"
    webvtt_file = f"./video_output/{video_no}/speech/subtitle.vtt"

    with open(output_file, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

    with open(webvtt_file, "w", encoding="utf-8") as file:
        file.write(submaker.generate_subs(sub_len))

async def generate_speech_split(video_no, input_text, voice = "en-US-AnaNeural") -> None:

    os.mkdir(f"../speech/{video_no}")
    texts = re.split(r'\.|\?|\!', input_text)
    #en-US-RogerNeural、en-US-AnaNeural
    
    output_file = f"./video_output/{video_no}/speech/speech.mp3"
    webvtt_file = f"./video_output/{video_no}/speech/subtitle.vtt"
    error_time = error_switcher[voice]
    """Main function"""
    
    #for text in TEXTS:

    pre_endtime = 0
    print("Generating speech sound.")

    for i, text in enumerate(texts, start=1):
        if i==len(texts):
            break
        
   
        submaker = edge_tts.SubMaker()
        communicate = edge_tts.Communicate(text, voice)
        print("processing" + str(i) + ":" + text)
        with open(output_file, "ab") as file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    #print((chunk["offset"], chunk["duration"]), chunk["text"])

                    submaker.create_sub((pre_endtime+chunk["offset"], chunk["duration"]), chunk["text"])
                    pre_offset = chunk["offset"]
                    pre_duration = chunk["duration"]
            #error_time是音檔結尾空白時間估計值
            pre_endtime += pre_offset + pre_duration + error_time 
            #print("endtime:" , pre_endtime)
        with open(webvtt_file, "a", encoding="utf-8") as file:
            subtitle = submaker.generate_subs()
            if(i!=1):
                lines = subtitle.split('\n')

                lines = lines[2:]

                modified_string = '\n'.join(lines)
                subtitle = modified_string

            file.write(subtitle)
    return pre_endtime
    


