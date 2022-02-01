#!/bin/bash

bash download_datasets.sh
bash download_models.sh

pip install -r requirements.txt


thresholds=(0.90 0.80 0.70 0.60 0.50 0.40 0.30 0.20 0.10)

echo testing_ig_mean_simple
for threshold in ${thresholds[@]}; do
  echo Testing threshold: $threshold -------------------------------
  python test_ig.py -t $threshold -wa mean -dn cei -mt 10 -m simple
done

echo testing_ig_mean_multi
for threshold in ${thresholds[@]}; do
  echo Testing threshold: $threshold -------------------------------
  python test_ig.py -t $threshold -wa mean -dn cei -mt 10 -m multi
done

thresholds=(0.80 0.70 0.60 0.50 0.40 0.30 0.20 0.10)

echo testing_ig_mean_abs_simple
for threshold in ${thresholds[@]}; do
  echo Testing threshold: $threshold -------------------------------
  python test_ig.py -t $threshold -wa mean_abs -dn cei -mt 10 -m simple
done

echo testing_ig_mean_abs_multi
for threshold in ${thresholds[@]}; do
  echo Testing threshold: $threshold -------------------------------
  python test_ig.py -t $threshold -wa mean_abs -dn cei -mt 10 -m multi
done

thresholds=(0.70 0.60 0.50 0.40 0.30 0.20 0.10)
echo testing_ig_mean_pos_multi
for threshold in ${thresholds[@]}; do
  echo Testing threshold: $threshold -------------------------------
  python test_ig.py -t $threshold -wa mean_pos -dn cei -mt 10 -m multi
done

echo finished