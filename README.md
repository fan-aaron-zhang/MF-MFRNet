# Overview:

 - `CNN`: contains code of MFRNet, training and evaluation.
 - `CNN/models/VTM162_BD_MFRNet_ORI`: contains MF-MFRNet models trained for 4 different QP settings.
 - `CNN/BC4/submit_train.sh`: BC4 job script for training MFRNet.
 - `EVAL_BC4.sh` : BC4 job script for decoding and evaluation.
 - `mainDec_EVAL.m`: decoder MATLAB script, called in the job script above.
 - `Dockerfile`: dockerfile used to generate the test environment on 3090 machines.


# Requirements:
Supposed to run on BC4. Load the tensorflow module by running
```
module add languages/anaconda3/2019.07-3.6.5-tflow-1.14
```
