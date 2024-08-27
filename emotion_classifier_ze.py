import warnings
warnings.filterwarnings('ignore')

from transformers import T5ForConditionalGeneration, AutoTokenizer, file_utils
import openai
import pandas as pd
import re
import csv
import time

API_KEY = 'YOUR_KEY'
client = openai.OpenAI(
    api_key = API_KEY,
)
FILE_NAME = 'gpt3.5_classifications_para1.csv'

context = "We are digital humanities researchers looking to code emotions within novels on the paragraph-by-paragraph level. Specifically, these paragraph describe places within the Lake District in England."
question = "What emotion does the following paragraph evoke?"
constraint = "Limit the number of distinct emotions to under 5, i.e. the four or less most prominent emotions. Many of these paragraphs will just be the formatting or titling before or after the true text. If this is the case, please note as such instead of categorizing."

df = pd.read_csv('final_paragraphs.csv')
response_arr = []
i = 0

print("Starting...")
for paragraph in df['paragraph'][i:]:
    start_time = time.time()
    i += 1  

    # Create the complete prompt
    prompt = f"{context}\n\n{question}\n\nParagraph: \"{paragraph}\"\n{constraint}"
    #print(prompt)
    
    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "user",
                "content": prompt
            },
        ],
        max_tokens=100,
        n=1,
    )

    response_str = response.choices[0].message.content
    #print(f'\nCHATGPT RESPONSE: \n{response_str}')
    response_arr.append(response_str)

    print(time.time() - start_time)
    if i % 100 == 0:
        print(i)
        print(response_arr[-10:])
        
        with open(FILE_NAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(response_arr)

with open(FILE_NAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(response_arr)

df['gpt_class'] = response_arr
df.to_csv('BBBB.csv', index=False)

'''
offload_folder = "../models"
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", torch_dtype=torch.float32, device_map="auto", offload_folder=offload_folder)                                                                 
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

context = "We are digital humanities researchers looking to code emotions within novels on the sentence-by-sentence level. Specifically, these sentences describe places within the Lake Distrinct in England"
question = "What emotion does the following sentence evoke? Choose one of the 28 options by picking a number."
sentence = "Oh, thrice happy moment, for Windermere's sheet In its bright, silvan beauty lay stretched at our feet!"
options = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiousity', 'desire',
           'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 
           'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'suprise', 'neutral']
constraint = "Even if you are uncertain, you must pick a number corresponding to an emotion without using any other numbers or words."

formatted_options = ""
for index, choice in enumerate(options, start=1):
    formatted_options += f'{index}. {choice}\n'

input_string = f'{context}\n{question}\n{sentence}\n\n{formatted_options}\n{constraint}'
print(input_string)

# If CUDA is not available, fallback to CPU
inputs = tokenizer(input_string, return_tensors="pt", return_attention_mask=True)

device = "cuda" if torch.cuda.is_available() else "mps"
model.to(device)
inputs.to(device)

outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=200)

print(tokenizer.decode(outputs[0]))
'''