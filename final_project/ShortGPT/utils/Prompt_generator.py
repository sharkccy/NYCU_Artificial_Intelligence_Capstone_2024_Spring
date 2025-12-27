import openai
import re
import os
def generate_prompt(video_no, paragraph, key):
    # openai.api_key = key
    client = openai.OpenAI(api_key=key)
    ## system_msg = "Step1: According to the given the paragraph, creates english scripts of a short video about 1 minute. The script should contain an engaging opening, core content or revelant data, and an ending that interact with the audience. Step2: output the word[Prompts for images]. Step3: Please generate prompts using for generating 10 images that are suitable for video content in step 1. Step 4: output the word[Prompts for BGM]. Step 5: Please Generate prompt for generating a background music that is suitable for the video in step 1. The prompt should consist of five tags such like music style, emotional feeling, tempo speed, etc."
    system_msg = (
        "According to the given paragraph/topic, create an English short story for a short video about 30~60 seconds. The script should contain an engaging opening marked as [Opening], core content or relevant data marked as [Core Content], and an ending that interacts with the audience marked as [Ending]. And please do not note narrator: before lines"
    )
    #paragraph = "腸病毒、流感等病毒近期持續發威，各大醫療院所人滿為患，其中人口超過400萬人的新北市不少醫院傳出一床難求，甚至有醫院的急診室等病床人數高達60人，對此新北衛生局表示，由於護理人力缺乏，因護病比有相關規定，有些醫院有床也開不了。新北衛生局專委楊時豪表示，「部分醫院的確因為人力上面調度，有時候剛好缺額，或是說還沒招聘到，為了符合護病比，所以有時候它會有些床就暫時沒有辦法收治病人。」"
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"given_paragraph: {paragraph}"},
        ],
        max_tokens = 4096
    )
    script = response.choices[0].message.content
    print(script)
    steps = re.split(r'\[Opening\]|\[Core Content\]|\[Ending\]', script)
    script_content = steps[1].strip() + steps[2].strip() + steps[3].strip()
    print(f"script content\n{script_content}")
    try:
        os.mkdir(f"./video_output/{video_no}/script")
        file = open(f"./video_output/{video_no}/script/script.txt", "w", encoding="utf-8")
        file.write(script_content)
        print("The script has been saved to script.txt")
    except Exception as e:
        print(e)
    finally:
        if file:
            file.close()
            print("Script file has been closed.")


    system_msg = (
        "Step1 : Please generate prompts for generating 10 images that are suitable for the video content with the given script. Marked as [Prompts for images]"
        "Step2 : Please generate prompts for generating 1 lofi style background music that are suitable for the given script. Prompts should contain style, Tempo, BPM and Emotion and Atmosphere. Marked as [Prompts for BGM]"
    )
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Given script: {script_content}"},
        ],
        max_tokens = 4096
    )
    prompts = response.choices[0].message.content
    print(prompts)

    steps = re.split(r'\[Prompts for images\]|\[Prompts for BGM\]', prompts)
    image_prompts = steps[1].strip()
    music_prompt = steps[2].strip()
    print(f"image prompt\n{image_prompts}")
    print(f"music_prompt\n{music_prompt}")

    try:
        file = open(f"./video_output/{video_no}/script/image_prompts.txt", "w", encoding="utf-8")
        file.write(image_prompts)
        print("The image prompts have been saved to image_prompts.txt")
    except Exception as e:
        print(e)
    finally:
        if file:
            file.close()
            print("Image prompt file has been closed.")

    try:
        file = open(f"./video_output/{video_no}/script/music_prompt.txt", "w", encoding="utf-8")
        file.write(music_prompt)
        print("The music prompt has been saved to music_prompt.txt")
    except Exception as e:
        print(e)
    finally:
        if file:
            file.close()
            print("Music prompt file has been closed.")
    return script_content, image_prompts, music_prompt

