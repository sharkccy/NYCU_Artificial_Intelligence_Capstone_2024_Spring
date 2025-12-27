from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os
import requests

driver = webdriver.Chrome('driver\chromedriver.exe')
url = 'https://www.ptt.cc/bbs/Stock/index.html'
driver.get(url)
time.sleep(1)
titleList = []

for i in range(3):
    elements = driver.find_elements(By.CLASS_NAME, "title")
    titleList.append([e.text for e in elements])
    link = driver.find_element(By.LINK_TEXT, "‹ 上頁")
    link.click()
    time.sleep(3)
print(titleList)
driver.close()
