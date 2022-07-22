#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:50:46 2020

@author: sk
"""
"""
import urllib.request

fp = urllib.request.urlopen("https://www.bet365.es/#/AC/B13/C1/D50/E2/F163/")
mybytes = fp.read()

mystr = mybytes.decode("utf8")
fp.close()

print(mystr)
"""


"""
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

br = webdriver.Firefox(executable_path=r'/home/sk/Desktop/bet/path/geckodriver')
br.get('https://www.bet365.es/#/AC/B13/C1/D50/E2/F163/')

"""
"""
save_me = ActionChains(br).key_down(Keys.CONTROL)\
         .key_down('s').key_up(Keys.CONTROL).key_up('s')
save_me.perform()
"""


url = 'https://www.bet365.es/#/AC/B13/C1/D50/E2/F163/'

#url = 'https://www.google.es'

#from selenium import webdriver
#from time import sleep
"""
driver = webdriver.Firefox(executable_path=r'/home/sk/Desktop/bet/path/geckodriver')
driver.get(url)

sleep(5)
html = driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
print(html)
"""


"""
path = '/home/sk/Desktop/bet/path/phantomjs-2.1.1-linux-x86_64/bin/phantomjs'
from selenium import webdriver
from time import sleep
driver = webdriver.PhantomJS(executable_path=path)
driver.get(url)
#sleep(5)
#html = driver.find_element_by_tag_name('html').get_attribute('innerHTML')
html = driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
"""

"""
import urllib3
from bs4 import BeautifulSoup
import json
import urllib.request

fp = urllib.request.urlopen("https://www.bet365.es/#/AC/B13/C1/D50/E2/F163/")


# read all data
page = fp.read()

# convert json text to python dictionary
data = json.loads(page)

print(data['principal_activities'])

"""
"""
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.expected_conditions import visibility_of_element_located
from selenium.webdriver.support.ui import WebDriverWait
import pyautogui
from time import sleep
import json

#url = 'https://www.google.es'
url = 'https://www.bet365.es/#/AC/B13/C1/D50/E2/F163/'


path = '/home/sk/Desktop/bet/path/phantomjs-2.1.1-linux-x86_64/bin/phantomjs'
#url = 'https://www.google.com/flights/explore/#explore;f=BDO;t=r-Asia-0x88d9b427c383bc81%253A0xb947211a2643e5ac;li=0;lx=2;d=2018-01-09'
#driver = webdriver.PhantomJS(executable_path=path)

driver = webdriver.Firefox(executable_path=r'/home/sk/Desktop/bet/path/geckodriver')

page = driver.get(url)

data1 = page.read()

# convert json text to python dictionary
data = json.loads(data1)

"""


# wait until results are loaded
#CCTAAACTATAGAAGGACAGCTCAAACACAAAGTTACCTAAACTATAGAAGGACAGCTCAAACACAAAGTTACCTAAACTATAGAAGGACAGCTCAAACACAAAGTTACCTAAACTATAGAAGGACAGCTCAAACACAAAGTTACCTAAACTATAGAAGGACA.html
#CCTAAACTATAGAAGGACAGCTCAAACACAAAGTTACCTAAACTATAGAAGGACAGCTCAAACACAAAGTTACCTAAACTATAGAAGGACAGCTCAAACACAAAGTTACCTAAACTATAGAAGGACAGCTCAAACACAAAGTTACCTAAACTATAGAAGGACA.html





"""
import webbrowser
webbrowser.open(url)

SEQUENCE = 'final'




#WebDriverWait(driver, 60).until(visibility_of_element_located((By.ID, 'grView')))
#sleep(1)
# open 'Save as...' to save html and assets

pyautogui.hotkey('ctrl', 's')
#time.sleep(3)
pyautogui.typewrite(SEQUENCE + '.html')
pyautogui.hotkey('enter')
pyautogui.hotkey('enter')

#time.sleep(2)

pyautogui.hotkey('alt', 'f4')

# osea ya entiende los comandos y los pulsa
# va pero como sabemos cuanto hay que esperar
# y el html cuando se descarga?

# mira, no tokes nada

#time.sleep(1)

#NO LO HE PROBADO

#file = open('NUEVO.html.html', "r") 
#print(file.read() )

# nose xke ta en esta carpeta, tendria ke estar el .py y el html a la altura


webbrowser.open('file:///home/sk/Desktop/tfg/final.html.html')


pyautogui.hotkey('ctrl', 'u')
time.sleep(1)


pyautogui.hotkey('ctrl', 'a')
time.sleep(1)
pyautogui.hotkey('ctrl', 'c') 

data = pyautogui.hotkey('ctrl', 'v') 

"""


