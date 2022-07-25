trainHR=.
trainLR=.
validBlocks=/delete_db/valid/
ratio=1
blockSize=96
resultsFolder=/
trainFolder=delete_db
subName=test
nlayers=16
nepochs=1
nepochs_GAN=0
paramLR_init=0.0001
paramLR_gan=0.0001
paramLD=0.1
decayEpochs=100
decayEpochsGAN=100
GAN=0
readBatch_flag=1
BN=0
Loss=L1
inputFormat=RGB
interpUV=nearest

python main_GAN_VGG_PSNR_MFRNet.py --mode=train --trainHR=${trainHR} --trainLR=${trainLR} --validBlocks=${validBlocks} --ratio=${ratio} --blockSize=${blockSize} --inputFormat=${inputFormat} --resultsFolder=${resultsFolder} --trainName=${trainFolder} --subName=${subName} --nlayers=${nlayers} --nepochs=${nepochs} --nepochs_GAN=${nepochs_GAN} --paramLR_INIT=${paramLR_init} --paramLR_GAN=${paramLR_gan} --paramLD=${paramLD} --decayEpochs=${decayEpochs} --decayEpochsGAN=${decayEpochsGAN} --GAN=${GAN} --readBatch_flag=${readBatch_flag} --BN=${BN} --loss=${Loss} --interpUV=${interpUV}

