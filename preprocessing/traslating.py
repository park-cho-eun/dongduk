import pandas as pd
import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import emoji

#한글리뷰->영어로 번역 by파파고
def translating(df, col):
    driver = webdriver.Chrome(executable_path="C:\\Users\\INFOSTAT-18\\Desktop\\project\\chromedriver")
    driver.get('https://papago.naver.com/')
    time.sleep(1)

    num_list = range(500, df.shape[0], 500)
    nums = range(1, (df.shape[0] // 500) + 2)

    for k, num in zip(nums, num_list):
        review_list = []
        title_list = []
        for row in tqdm.tqdm(range(num-500, num)):
            try:
                rev = df[col][row]

                driver.find_element(By.XPATH, "/html/body/div/div/div[1]/section/div/div[1]/div[1]/div/div[3]/label/textarea").send_keys(rev)
                driver.find_element(By.XPATH, "/html/body/div/div/div[1]/section/div/div[1]/div[1]/div/div[4]/div/button").click()
                time.sleep(2.5)
                txt = driver.find_element(By.XPATH, "/html/body/div/div/div[1]/section/div/div[1]/div[2]/div/div[5]/div").text
                review_list.append(txt)
                time.sleep(1.5)
                driver.find_element(By.XPATH, "/html/body/div/div/div[1]/section/div/div[1]/div[1]/div/div[3]/label/textarea").clear()
                time.sleep(1.5)

                if type(reviews['title'][row]) == 'str.':
                    tit = reviews['title'][row]
                    driver.find_element(By.XPATH, "/html/body/div/div/div[1]/section/div/div[1]/div[1]/div/div[3]/label/textarea").send_keys(tit)
                    driver.find_element(By.XPATH, "/html/body/div/div/div[1]/section/div/div[1]/div[1]/div/div[4]/div/button").click()
                    time.sleep(2.5)
                    ti = driver.find_element(By.XPATH, "/html/body/div/div/div[1]/section/div/div[1]/div[2]/div/div[5]/div").text
                    title_list.append(ti)
                    time.sleep(1.5)
                    driver.find_element(By.XPATH, "/html/body/div/div/div[1]/section/div/div[1]/div[1]/div/div[3]/label/textarea").clear()
                    time.sleep(1.5)
                else:
                    title_list.append('0')
            except:
                review_list.append('1')
                title_list.append('1')

        ser = pd.DataFrame(review_list, columns=['Text_en'])
        ser.to_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\watcha_en_'+str(k)+'.csv')
        print('총 '+str(df.shape[0])+'중에 '+str(k)+'번째 끝')

#이모지 별도 처리
for row in tqdm.tqdm(range(df.shape[0])):
  df['Text'][row] = emoji.demojize(df['Text'][row])

df.to_csv('/content/drive/MyDrive/dd/twt/watcha.csv')