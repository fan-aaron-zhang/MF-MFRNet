# Overview:

 - `CNN`: contains code of MFRNet, training and evaluation.
 - `CNN/BC4/submit_train.sh`: BC4 job script for training MFRNet.
 - `EVAL_BC4.sh` : BC4 job script for decoding and evaluation.
 - `mainDec_EVAL.m`: decoder MATLAB script, called in the job script above.
 - `Dockerfile`: dockerfile used to generate the test environment on 3090 machines.


# Requirements:
Supposed to run on BC4. Load the tensorflow module by running
```
module add languages/anaconda3/2019.07-3.6.5-tflow-1.14
```


# Training MFRNet:

See `CNN/BC4/submit_train.sh`. In the job script, working directory should be `MFRNet/CNN`, and the `main_GAN_VGG_PSNR_MFRNet.py` python script should be called with arguments specified. See below the description of the arguments specified in the BC4 training job script (in `CNN/BC4/submit_train.sh`):

- `trainHR`: not used during training, can be set to anything.
- `trainLR`: not used during training, can be set to anything.
- `validBlocks`: the **absolute path** of the directory that contains validation data. For example, if `validBlocks='work/valid'`, then inside the folder the structure should be `work/valid/HR` and `work/valid/LR`, i.e. `work/valid` should contain two folders `HR/` and `LR/`.
- `ratio`: always set to 1 for PP.
- `blockSize`: spatial size of training image blocks, set to 96.
- `resultsFolder`: the **absolute path** of the directory that contains the training data folder (`trainFolder` below). That is, training data should be located in `resultsFolder/trainFolder`.
- `trainFolder`: the **name of the folder** that contains training data, inside which the structure is `trainFolder/train_dataset/batch_blocks/HR`, `trainFolder/train_dataset/batch_blocks/HR`.
- `subName`: name of current training settings, can be given by the user.
- `nlayers`: number of Denselayers in the network. Keep as 16.
- `nepochs`: number of training epochs.
- `paramLR_init`: initial learning rate. 1e-4 is good.
- `paramLR_gan`: ignore, because not using GAN.
- `paramLD`: ignore as well.
- `decayEpochs`: number of training epochs until learning rate decay.
- `decayEpochsGAN`: ignore.
- `GAN`: whether or not using GAN. Set to 0.
- `readBatch_flag`: whether reading batches directly. Set to 1.
- `BN`: batch normalisation. Set to 0 in original implementation.
- `Loss`: loss function, set to 'L1'.
- `inputFormat`: set to 'RGB'.
- `interpUV`: set to 'nearest'
