from moviepy.editor import ImageClip, CompositeVideoClip, CompositeAudioClip, VideoFileClip, AudioFileClip, TextClip
from moviepy.video.tools.subtitles import SubtitlesClip
from mutagen.mp3 import MP3
from moviepy.config import change_settings
import os, re

record = 0

def generate_video(video_no, transition_duration=1, fps = 24):
    change_settings({"use_ffmpeg_gpu_decoder":True})
    img_directory = f"./video_output/{video_no}/image"
    output_directory = f"./video_output/{video_no}"
    bgm_dir = f"./video_output/{video_no}/bgm/bgm.wav"
    speaking_dir= f"./video_output/{video_no}/speech/speech.mp3"
    subtitles_dir = f"./video_output/{video_no}/speech/subtitle.vtt"

    display_duration = MP3(speaking_dir).info.length/len(os.listdir(img_directory))

    clips = []
    for img in os.listdir(img_directory):
        if img.endswith('.jpg'):
            path = os.path.join(img_directory, img)
            clip = ImageClip(path, duration=display_duration + transition_duration)
            clip = clip.crossfadeout(transition_duration)
            clips.append(clip) 
    
    video_fx_list = [clips[0]]
    index = clips[0].duration - transition_duration
    for video in clips[1:]:
        video_fx_list.append(video.set_start(index).crossfadein(transition_duration))
        index += video.duration - transition_duration
    video = CompositeVideoClip(video_fx_list)
    video = add_audio_to_video(video, bgm_dir, speaking_dir)
    video = add_subtitles_to_video(video, subtitles_dir)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  
    output_filename = os.path.join(output_directory, f"{video_no}.mp4")
    video.write_videofile(output_filename, codec='libx264', fps=fps)
    print(f"Video saved at {output_filename}")

def add_audio_to_video(video, bgm_path, speaking_path):
    bgm = AudioFileClip(bgm_path).audio_loop(duration=video.duration).volumex(0.5)
    speaking = AudioFileClip(speaking_path).audio_loop(duration=video.duration)
    video = video.set_audio(CompositeAudioClip([bgm, speaking]))
    return video
    # video.write_videofile(output_path, codec='libx264', fps=24)
    # print(f"Video saved at {output_path}")

def convert_time_to_seconds(timestr):
    hours, minutes, seconds = timestr.split(':')
    seconds, milliseconds = seconds.split('.')
    milliseconds = float(milliseconds) / 1000
    total_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds) + milliseconds
    return total_seconds

def parse_vtt(subtitles_dir):
    global record  # 宣告 record 為全域變數
    with open(subtitles_dir, 'r', encoding='utf-8') as file:
        content = file.read()
    subtitles = []
    # Regex to find timestamps and text
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.*?)(?=\n\d{2}:\d{2}:\d{2}\.\d{3} -->|\Z)', re.DOTALL)
    for match in re.finditer(pattern, content):
        start = match.group(1)
        end = match.group(2)
        text = match.group(3)
        # print(start,end,text)
        # Calculate duration in seconds
        start_sec = record
        ## convert_time_to_seconds(start)
        end_sec = convert_time_to_seconds(end) + 0.5
        record = end_sec
        duration = end_sec - start_sec
        subtitles.append((start_sec, end_sec, text, duration))
        # print(subtitles)
    return subtitles

def wrap_text(text, width):
    lines = []
    while len(text) > width:
        space_index = text.rfind(' ', 0, width)
        if space_index == -1:
            space_index = width
        lines.append(text[:space_index])
        text = text[space_index:].strip()
    lines.append(text)
    return "\n".join(lines)

def text_appearance(subtitle):
    wrapped_text = wrap_text(subtitle[2], 35)  # 假設每行不超過35個字元
    return TextClip(wrapped_text, font="ArchivoBlack",
                    fontsize=108,
                    color='white',
                    stroke_color='black',
                    stroke_width=3,
                    size=(800, 1920),
                    method='caption').set_start(subtitle[0]).set_end(subtitle[1]).set_position(('center', 'center'))

def add_subtitles_to_video(video, subtitles_dir):
    clips = [video] + [text_appearance(subtitle) for subtitle in parse_vtt(subtitles_dir)]
    video = CompositeVideoClip(clips)
    return video




