#!/bin/bash

echo Testing lrp_mean_pos -------------------------------
python test_lrp.py -wa mean_pos -dn HoC -mt 10 -m simple

echo Testing lrp_mean_pos_multi -------------------------------
python test_lrp.py -wa mean_pos -dn HoC -mt 10 -m multi

echo Testing lrp_mean -------------------------------
python test_lrp.py -wa mean -dn HoC -mt 10 -m simple

echo Testing lrp_mean_multi -------------------------------
python test_lrp.py -wa mean -dn HoC -mt 10 -m multi

echo Testing lrp_mean_abs -------------------------------
python test_lrp.py -wa mean_abs -dn HoC -mt 10 -m simple

echo Testing lrp_mean_abs_multi -------------------------------
python test_lrp.py -wa mean_abs -dn HoC -mt 10 -m multi