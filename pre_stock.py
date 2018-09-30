#!/usr/bin/python
#-*- coding:utf-8 -*-

import pandas as pd

from random import uniform
from time import sleep

code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0] # 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌 code_df.종목코드 = code_df.종목코드.map('{:06d}'.format) # 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다. code_df = code_df[['회사명', '종목코드']] # 한글로된 컬럼명을 영어로 바꿔준다. code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'}) code_df.head()

# 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌
code_df.종목코드 = code_df.종목코드.map('{:06d}'.format)

# 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다.
code_df = code_df[['회사명', '종목코드']]

# 한글로된 컬럼명을 영어로 바꿔준다.
code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'})
code_df.head()

# 종목 이름을 입력하면 종목에 해당하는 코드를 불러와
# 네이버 금융(http://finance.naver.com)에 넣어줌
def get_url(code):
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)

    print("URL = {}".format(url))
    return url

# 신라젠의 일자데이터 url 가져오기
item_name='신라젠'
code = code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
url = get_url(code)

# 일자 데이터를 담을 df라는 DataFrame 정의
df = pd.DataFrame()

# 1페이지에서 20페이지의 데이터만 가져오기
for page in range(1, 21):
    sleep(uniform(0.1, 1.0))
    pg_url = '{url}&page={page}'.format(url=url, page=page)
    df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)

# df.dropna()를 이용해 결측값 있는 행 제거
df = df.dropna()

df = df.rename(columns= {'날짜': 'Date', '종가': 'Close', '전일비': 'Diff', '시가': 'Open', '고가': 'High', '저가': 'Low', '거래량': 'Volume'})

df[['Close', 'Diff', 'Open', 'High', 'Low', 'Volume']] = df[['Close', 'Diff', 'Open', 'High', 'Low', 'Volume']].astype(float)

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Date'], ascending=True)

# Sort columns to Open, High, Low, Volume, Close
df = df.drop(columns=['Diff'])
cols = df.columns.tolist()
cols = cols[0:1] + cols[2:6] + cols[1:2]
df = df[cols]

print(df.head())

df.to_csv("stock/" + code + "_stock.csv", index=False)
