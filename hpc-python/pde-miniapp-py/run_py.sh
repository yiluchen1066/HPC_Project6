#!/bin/bash

fname="1node__weak_mem1024"
lscpu | grep "Model name" | tee $fname.data
echo -e "size\tnp\tNewton\tCG\ttime\titers_cg/timespent" | tee -a $fname.data

n=128
while [ $n -le 1024 ]
do
    p=1
    while [ $p -le 64 ]
    do
        mpirun -np $p python3 main.py $n 100 0.005 | tee -a $fname.data
        ((p*=4))
    done 
    ((n*=2))
done

