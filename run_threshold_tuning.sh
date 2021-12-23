#!/bin/bash

thresholds=(0.90 0.80 0.70 0.60 0.50 0.40 0.30 0.20 0.10)

for threshold in ${thresholds[@]}; do
  echo Testing percentile threshold: $threshold -------------------------------
  python test_lrp.py -t $threshold -wa mean_pos
done
