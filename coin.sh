#!/usr/bin/env bash

# Collect coin data
node coin.js ETH

# Estimate coin
python3 estimate.py ETH