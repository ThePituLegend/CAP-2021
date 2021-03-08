#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive

gcc -Ofast -std=c99 -lm common.c nn-main.c -o practica.x
./practica.x
