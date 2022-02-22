#!/bin/bash

echo Testing ig_mean_pos -------------------------------
python test_ig.py -wa mean_pos -dn HoC -mt 10 -m simple

echo Testing ig_mean_pos_multi -------------------------------
python test_ig.py -wa mean_pos -dn HoC -mt 10 -m multi

echo Testing ig_mean -------------------------------
python test_ig.py -wa mean -dn HoC -mt 10 -m simple

echo Testing ig_mean_multi -------------------------------
python test_ig.py -wa mean -dn HoC -mt 10 -m multi

echo Testing ig_mean_abs -------------------------------
python test_ig.py -wa mean_abs -dn HoC -mt 10 -m simple

echo Testing ig_mean_abs_multi -------------------------------
python test_ig.py -wa mean_abs -dn HoC -mt 10 -m multi