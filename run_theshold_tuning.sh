#!/bin/bash

thresholds=(90 80 70 60 50 40 30 20 10)

for threshold in ${thresholds[@]}; do
  echo Testing percentile threshold: $threshold -------------------------------
  python test_lrp.py -t $threshold
done
