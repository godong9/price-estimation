#!/usr/bin/env bash

# SKT, Sinhan, Naver
stocks=(017670 055550 035420)
for stock in ${stocks[*]}
do
    # Collect stock data
    python3 stock.py $stock

    # Estimate stock
    python3 estimate.py $stock

    # Collect coin data
    python3 coin.py ETH

    # Estimate coin
    python3 estimate.py ETH
done
