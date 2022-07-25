#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -o "/mnt/storage/scratch/eexfz/CNN/TEST_JVET_HM1620_BD9bit_QP32.o"

###################################################################################################
#### This code was developed by Mariana Afonso, Phd student @ University of Bristol, UK, 2018 #####
################################## All rights reserved © ##########################################

module add languages/anaconda3/3.5-4.2.0-tflow-1.7

which python

cd /mnt/storage/home/eexfz/CNN_ViSTRA

mode=evaluate
validHR=/mnt/storage/scratch/eexfz/JVET_SDR_10bit
validLR=/mnt/storage/scratch/eexfz/CNN/TEST_JVET_HM1620_BD9bit_QP32
resultsFolder=/mnt/storage/scratch/eexfz/CNN
trainName=TRAINING_DB18_HM1620_BD9bit_QP32
subName=YUV_L1
testEpoch=200
ratio=1
nlayers=16
GAN=0
nframes=0
eval_inputType=YUV
readBatch_flag=1
inputFormat=RGB

python main.py --mode=${mode} --validHR=${validHR} --validLR=${validLR} --resultsFolder=${resultsFolder} --trainName=${trainName} --subName=${subName} --testEpoch=${testEpoch} --ratio=${ratio} --nlayers=${nlayers} --GAN=${GAN} --nframes=${nframes} --eval_inputType=${eval_inputType} --readBatch_flag=${readBatch_flag} --inputFormat=${inputFormat} 
