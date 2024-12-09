import openai 
import json
import os
import tqdm
import random
import base64

with open('api.key') as f:
    openai.api_key = f.read().strip()

client = openai.OpenAI(
  api_key=openai.api_key,
  base_url="https://cmu.litellm.ai",
)

new = 'outputs/MeaCap_randval2014_memory_coco_lmTrainingCorpus_CBART_COCO_0.1_0.8_0.2_k200.json'
old = 'outputs/MeaCap_randval2014_memory_coco_lmTrainingCorpus_CBART_COCO_0.1_0.8_0.2_k200_og.json'

with open(new, 'r') as f:
    new_data = json.load(f)

with open(old, 'r') as f:
    old_data = json.load(f) 

def create_prompt(captions):
    prompt = f"""
You are given an image and two captions describing it. 
Your task is to evaluate which caption better describes the image.
Your critera is semantic richness, descriptiveness, and correctness.

Caption A: {captions[0][0]}
Caption B: {captions[1][0]}

Please state which caption is better (Caption A or Caption B). 
Simply state 'Caption A' if Caption A is better, otherwise 'Caption B'. 
"""
    return prompt

new = 0
old = 0

for img_name in tqdm.tqdm(old_data):
    img = base64.b64encode(open(f'data/coco/karpathy_test/images/randval2014/{img_name}.jpg', 'rb').read()).decode('ascii')
    new_caption = new_data[img_name][1]
    old_caption = old_data[img_name]
    captions = [(new_caption, 'new'), (old_caption, 'old')]
    random.shuffle(captions)
    prompt = create_prompt(captions)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}",
                    }
                }
            ]
        }]

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        n=1
    )

    text = response.choices[0].message.content
    if 'Caption A' in text:
        if captions[0][1] == 'new':
            new += 1
        else:
            old += 1
    else:
        if captions[1][1] == 'new':
            new += 1
        else:
            old += 1
        
print(f'new chosen proportion: {new/(new+old)}')
print(f'old chosen proportion: {old/(new+old)}')

