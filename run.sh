#!/bin/bash

for i in {1..2}
do
    log_file="logs/log_vanilla_$i.txt"
    python torch_vertical_FL_train.py --epochs 100 --model_type 'vertical' --organization_num 4 > "$log_file"
done