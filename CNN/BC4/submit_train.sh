#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --output=/user/work/hw22082/out_train/%A_%a.out

###################################################################################################
#### This code was developed by Mariana Afonso, Phd student @ University of Bristol, UK, 2018 #####
################################## All rights reserved © ##########################################

module add languages/anaconda3/2019.07-3.6.5-tflow-1.14

which python

QP_array=('16' '21' '26' '31')

cd /user/home/hw22082/VTM16.2_HDR_EBDA/MFRNet/CNN/

trainHR=.
trainLR=.
validBlocks="/user/work/hw22082/CLIC_Videos/VAL_QP${QP_array[$SLURM_ARRAY_TASK_ID]}/train_dataset/batch_blocks"
ratio=1
blockSize=96
resultsFolder="/user/work/hw22082/CLIC_Videos"
#trainFolder="TRAINING_QP${QP_array[$SLURM_ARRAY_TASK_ID]}"
trainFolder="NOHDR_QP${QP_array[$SLURM_ARRAY_TASK_ID]}"
subName="newNOHDR_OF_QP${QP_array[$SLURM_ARRAY_TASK_ID]}"
nlayers=16
nepochs=200
nepochs_GAN=0
paramLR_init=0.0001
paramLR_gan=0.0001
paramLD=0.1
decayEpochs=8
decayEpochsGAN=100
GAN=0
readBatch_flag=1
BN=0
Loss=L1
inputFormat=RGB
interpUV=nearest

python main_GAN_VGG_PSNR_MFRNet.py --mode=train --trainHR=${trainHR} --trainLR=${trainLR} --validBlocks=${validBlocks} --ratio=${ratio} --blockSize=${blockSize} --inputFormat=${inputFormat} --resultsFolder=${resultsFolder} --trainName=${trainFolder} --subName=${subName} --nlayers=${nlayers} --nepochs=${nepochs} --nepochs_GAN=${nepochs_GAN} --paramLR_INIT=${paramLR_init} --paramLR_GAN=${paramLR_gan} --paramLD=${paramLD} --decayEpochs=${decayEpochs} --decayEpochsGAN=${decayEpochsGAN} --GAN=${GAN} --readBatch_flag=${readBatch_flag} --BN=${BN} --loss=${Loss} --interpUV=${interpUV}
