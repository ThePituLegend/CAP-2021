#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive

make
perf stat ./neuralPar.x