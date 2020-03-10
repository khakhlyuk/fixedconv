#!/bin/bash

c=2
ff=n
for model in fixed_resnet20 fixed_resnet32
do
  for k in $(seq 1 3)
  do
    for conv_type in G A M
    do
      echo "python -u train_resnet.py --model=$model --ff=$ff -k=$k --conv_type=$conv_type -c=$c"
      python -u train_resnet.py --model=$model --ff=$ff -k=$k --conv_type=$conv_type -c=$c
    done
  done
done

