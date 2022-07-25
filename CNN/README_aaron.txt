###################################################################################################
#### This code was developed by Mariana Afonso, Phd student @ University of Bristol, UK, 2018 #####
################################## All rights reserved © ##########################################

Instructions:

- Install Pycharm: there is a free version. Good for debugging.
- Install python 3.5 (I have 3.5.2) or 3.6 + libraries + Tensorflow gpu (I have 1.8.0)
- Python dependencies: Tensorlayer, numpy, scipy, matplotlib, sklearn and imageio 

Quick uses (examples):

- Create training dataset:

--trainHR=M:\Aaron\ViSTRA\TRAINING_ORIG\DB18_10bit
--trainLR=M:\Aaron\ViSTRA\TRAINING_RESULTS\HM1620_TRAINING_DB18_9bit\REC\QP37
--validBlocks=.
--resultsFolder=E:\work\data\CNN\
--trainName=TRAINING_DB18_HM1620_BD9bit_QP37
--subName=useless
--blockSize=96
--ratio=1
--rotate=0
--readBatch_flag=1

- Create validation dataset:

--trainHR=M:\Aaron\ViSTRA\JVET_CTC\SDR_10bit
--trainLR=M:\Aaron\ViSTRA\JVET_RESULTS\JVET_9bit_HM1620_RA\REC\QP37
--validBlocks=.
--resultsFolder=E:\work\data\CNN\
--trainName=VALIDATION_JVET_HM1620_BD9bit_QP37
--subName=useless
--blockSize=96
--ratio=1
--rotate=0
--readBatch_flag=1
--createValid=1

- Run YUV model training

--trainHR=.
--trainLR=.
--validBlocks=E:\work\data\CNN\VALIDATION_JVET_HM1620_BD9bit_QP37\train_dataset\batch_blocks
--resultsFolder=E:\work\data\CNN\
--trainName=TRAINING_DB18_HM1620_BD9bit_QP37
--subName=BD_L1_Y
--blockSize=96
--ratio=1
--nlayers=16
--nepochs=200
--nepochs_GAN=0
--paramLR_INIT=0.0001
--paramLR_GAN=0.0001
--decayEpochs=100
--decayEpochsGAN=100
--paramLD=0.1
--GAN=0
--readBatch_flag=1
--inputFormat=RGB
--BN=0
--loss=L1
--interpUV=nearest

- Run YUV model evaluation

--mode=evaluate
--evalHR=M:\Aaron\ViSTRA\TRAINING_RESULTS\HM1620_TRAINING_DB18_9bit\REC\QP27\Aamerican-football-scene4_3840x2160_60fps_10bit_420_qp21_BD09.yuv
--evalLR=M:\Aaron\ViSTRA\TRAINING_RESULTS\HM1620_TRAINING_DB18_9bit\REC\QP27\Aamerican-football-scene4_3840x2160_60fps_10bit_420_qp21_BD09.yuv
--testModel=0
--ratio=1
--nlayers=16
--GAN=0
--nframes=1
--eval_inputType=YUV
--readBatch_flag=1
--inputFormat=RGB