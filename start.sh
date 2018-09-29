#!/usr/bin/env bash

# SKT, Sinhan, Naver
stocks=(017670.KS 055550.KS 035420.KS)
for stock in ${stocks[*]}
do
    # Collect stock data
    python3 collect.py $stock

    # Estimate stock
    python3 estimate.py $stock
done


