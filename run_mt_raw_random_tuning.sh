#!/bin/bash

echo Testing multi_task -------------------------------
python test_multi_task.py -dn HoC -mt 10

echo Testing random -------------------------------
python test_random.py -dn HoC -mt 10 -m simple

echo Testing random_multi_task -------------------------------
python test_random.py -dn HoC -mt 10 -m multi

echo Testing raw_attn_0_6 -------------------------------
python test_raw_attn.py -dn HoC -mt 10 -sl 0 -ll 6 -m simple

echo Testing raw_attn_0_6_multi -------------------------------
python test_raw_attn.py -dn HoC -mt 10 -sl 0 -ll 6 -m multi

echo Testing raw_attn_6_12 -------------------------------
python test_raw_attn.py -dn HoC -mt 10 -sl 6 -ll 12 -m simple

echo Testing raw_attn_6_12_multi -------------------------------
python test_raw_attn.py -dn HoC -mt 10 -sl 6 -ll 12 -m multi

echo Testing raw_attn_0_12 -------------------------------
python test_raw_attn.py -dn HoC -mt 10 -sl 0 -ll 12 -m simple

echo Testing raw_attn_0_12_multi -------------------------------
python test_raw_attn.py -dn HoC -mt 10 -sl 0 -ll 12 -m multi