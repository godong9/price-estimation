#!/usr/bin/python

import sys
import fix_yahoo_finance as yf

from datetime import datetime

stock = sys.argv[1]

print("Stock:", stock)

start = "2000-01-01"
end =  datetime.now().strftime("%Y-%m-%d")

print("start:", start)
print("end:", end)

data = yf.download(stock, start=start, end=end)

data.to_csv("stock/" + stock + "_stock.csv")