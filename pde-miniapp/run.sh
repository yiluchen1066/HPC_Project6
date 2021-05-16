#!/bin/bash

fname="1node_mem1024"
lscpu | grep "Model name" | tee $fname.data
echo -e "size\tnp\tNewton\tCG\ttime\iters_cg/timespent" | tee -a $fname.data

n=128
while [ $n -le 128 ]
do
    p=1
    while [ $p -le 32]
    do
        mpirun -np $p ./main $n 100 0.005 | tee -a $fname.data
        ((p+=1))
    done 
    ((n*=2))
done