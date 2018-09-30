#!/usr/bin/python
#-*- coding:utf-8 -*-

import sys
import pandas as pd

from random import uniform
from time import sleep

stock = sys.argv[1]

print("========== [Collect] start! ==========")

print("Stock:", stock)

# 종목 이름을 입력하면 종목에 해당하는 코드를 불러와
# 네이버 금융(http://finance.naver.com)에 넣어줌
def get_url(code):
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)

    print("URL = {}".format(url))
    return url

url = get_url(stock)

# 일자 데이터를 담을 df라는 DataFrame 정의
df = pd.DataFrame()

# 1페이지에서 400페이지의 데이터 가져오기
for page in range(1, 400):
    sleep(uniform(0.1, 0.5))
    pg_url = '{url}&page={page}'.format(url=url, page=page)
    print("Data url = {}".format(pg_url))
    df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)

# df.dropna()를 이용해 결측값 있는 행 제거
df = df.dropna()

df = df.rename(columns= {'날짜': 'Date', '종가': 'Close', '전일비': 'Diff', '시가': 'Open', '고가': 'High', '저가': 'Low', '거래량': 'Volume'})

df[['Close', 'Diff', 'Open', 'High', 'Low', 'Volume']] = df[['Close', 'Diff', 'Open', 'High', 'Low', 'Volume']].astype(int)

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Date'], ascending=True)

# Sort columns to Open, High, Low, Volume, Close
df = df.drop(columns=['Diff'])
cols = df.columns.tolist()
cols = cols[0:1] + cols[2:6] + cols[1:2]
df = df[cols]

print(df.head())

df.to_csv("data/" + stock + ".csv", index=False)

print("========== [Collect] complete! ==========")
