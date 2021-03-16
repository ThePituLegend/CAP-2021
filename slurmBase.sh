#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive

make
perf record ./neuralBase.x