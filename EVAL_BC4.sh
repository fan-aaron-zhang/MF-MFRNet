#!/bin/bash
#SBATCH --job-name=noHDRBD_VTM162
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=1,5,40
#SBATCH --output=/user/work/hw22082/out_test/%A_%a.out

module add languages/anaconda3/2019.07-3.6.5-tflow-1.14
module add apps/matlab/2018a


which python

echo "Decoding start!"
scriptPath='/user/home/hw22082/VTM16.2_HDR_EBDA/MFRNet/'
hostDecoder='VTM1620_Decoder'
streamPath='/user/work/hw22082/JVET9bit/STREAM/'
yuvPath='/user/work/hw22082/JVET9bit/REC/'
statPath='/user/work/hw22082/JVET9bit/STAT/'
logPath='/user/work/hw22082/JVET9bit/'
origPath='/user/work/hw22082/Sequence_HDR/JVET_HDR_10bit/'


echo "Decode: sequence #${SLURM_ARRAY_TASK_ID} on NODEID: #${SLURM_NODEID}"
echo "SLURM_NODE_ALIASES: #${SLURM_NODE_ALIASES}"

time matlab -r "addpath(genpath('$scriptPath')); mainDec_EVAL('${scriptPath}','${hostDecoder}','${streamPath}','${yuvPath}','${origPath}','${statPath}',$SLURM_ARRAY_TASK_ID); exit;"