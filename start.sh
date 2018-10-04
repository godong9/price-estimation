#!/usr/bin/env bash

# SKT, Sinhan, Naver, LG Chem, CGV, JYP
stocks=(017670 055550 035420 051910 079160 035900)
for stock in ${stocks[*]}
do
    # Collect stock data
    python3 stock.py $stock

    # Estimate stock
    python3 estimate.py $stock
done
