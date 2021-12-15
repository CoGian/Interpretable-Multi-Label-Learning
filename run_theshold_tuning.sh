#!/bin/bash

thresholds=(10 20 30 40 50 60 70 80 90)

for threshold in ${thresholds[@]}; do
  echo Testing percentile threshold: $threshold -------------------------------
  python test_ig.py -t $threshold
done
