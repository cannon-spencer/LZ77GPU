#!/bin/bash
#SBATCH --job-name=GPU_LZ77            # Job name
#SBATCH --mail-type=END,FAIL                   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yu.hong@ufl.edu     # Where to send mail
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1                          # Run on a single A100 GPU
#SBATCH --mem=1gb                              # Job memory request
#SBATCH --time=00:10:00                        # Time limit hrs:min:sec
#SBATCH --output=GPU_SA_libcubwt%j.log         # Standard output and error log

# General Information
pwd; hostname; date

# run the program
./build/lz77_gpu /blue/boucher/share/LZ77GPU/libcubwt-1.0.0/test.txt


