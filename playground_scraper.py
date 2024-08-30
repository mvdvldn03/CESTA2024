from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
import csv

FILE_NAME = 'gpt4_classifications_paragraph.csv'

chrome_options = Options()
service = Service('/Users/maxv/Documents/Spatial Narratives/code/chromedriver')  # Replace with your chromedriver path

driver = webdriver.Chrome(service=service, options=chrome_options)
screen_width = driver.execute_script("return window.screen.width;")
screen_height = driver.execute_script("return window.screen.height;")
window_width = int(screen_width / 3)
window_height = screen_height
window_position_x = screen_width - window_width
window_position_y = 0  

driver.set_window_size(window_width, window_height)
driver.set_window_position(window_position_x, window_position_y)

driver.get("https://google.com")
input("Press Enter to once logged into https://aiplayground-prod.stanford.edu...")

'''
During this period, while the window is open, you will need to manually log into the aiplayground-prod.stanford.edu.
This could be automated later, but it would take some time with the validation process.
'''

df = pd.read_csv('final_paragraphs.csv')
response_arr = []
i = 0

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

    #Input the complete prompt into the AI Playground
    message_box = driver.find_element(By.ID, "prompt-textarea")
    message_box.send_keys(context)
    message_box.send_keys(Keys.SHIFT, Keys.ENTER)
    time.sleep(0.5)
    message_box.send_keys(Keys.SHIFT, Keys.ENTER)
    message_box.send_keys(question)
    message_box.send_keys(Keys.SHIFT, Keys.ENTER)
    time.sleep(0.5)
    message_box.send_keys(Keys.SHIFT, Keys.ENTER)
    message_box.send_keys(f'Paragraph: \"{paragraph}"\"')
    message_box.send_keys(Keys.SHIFT, Keys.ENTER)
    time.sleep(0.5)
    message_box.send_keys(Keys.SHIFT, Keys.ENTER)
    message_box.send_keys(constraint)                   

    message_box.send_keys(Keys.RETURN)
    
    #Wait for fully-fleshed response

    #Wait for start of answer creation
    while len(driver.find_elements(By.CSS_SELECTOR, ".markdown.prose")) % 2 != 0:
        time.sleep(1)
    
    #Wait for full answer creation
    time.sleep(10)
    response = driver.find_elements(By.CSS_SELECTOR, ".markdown.prose")[-1]

    print(response.text)
    print("\n")
    response_arr.append(response)

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
df.to_csv('full_GPT4-pg.csv', index=False)