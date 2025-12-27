from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import os
import requests

driver = webdriver.Chrome('driver\chromedriver.exe')
keywords = ['Tesla Semi']
keywords_2 = ['Ford f150', 
              'Ford mustang',
              'Ford bronco',
              'Ford expedition',
              'Ferrari 488pista',
              'Ferrari Purosangue',
              'Ferrari 812',
              'Ferrari F40',
              'Mercedes SL',
              'Mercedes G-wagon',
              'Mercedes Metris',
              'Mercedes S-class',
              'Tesla Model S',
              'Tesla Cybertruck',
              'Tesla Semi',
              'Tesla Model X',
              'Rolls Royce Dawn',
              'Rolls Royce Cullinan',
              'Rolls Royce Phantom',
              'Rolls Royce Spectre',]
for keyword in keywords:
    if os._exists(f"{keyword}") == False:
        os.mkdir(f"{keyword}")
    url = f'https://www.google.com/search?hl=en&tbm=isch&q={keyword} 
            1920x1080 -site:https://cars.usnews.com/cars-trucks'
    driver.get(url)
    driver.maximize_window()
    time.sleep(2)
    for i in range(3):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.randint(2, 5))

    image_elements = driver.find_elements(By.CLASS_NAME, 'rg_i')
    for i, image in enumerate(image_elements):
        try:
            image.click()  
            high_res_image = WebDriverWait(driver, 5).until(EC.visibility_of_element_located((By.CLASS_NAME, 'sFlh5c.pT0Scc.iPVvYb')))
            print(high_res_image)
            image_url = high_res_image.get_attribute('src') 
            # print(image_url)
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(os.path.join(f"{keyword}", f"{keyword}_{i}.jpg"), 'wb') as file:
                    file.write(response.content)
        except Exception as e:
            print(f"Error downloading image: {e}")
            print(f"Failed to download image with URL: {image}")
    driver.quit()

