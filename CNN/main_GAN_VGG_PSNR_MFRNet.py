#! /usr/bin/python
# -*- coding: utf8 -*-

###################################################################################################
#### This code was developed by Di Ma, Phd student @ University of Bristol, UK, 2019 #####
################################## All rights reserved © ##########################################

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import tensorflow as tf
import tensorlayer as tl
from model_NewSR_ESRGAN_New_V2 import *
from utils_NewSR_ESRGAN import *
import ntpath
import sys
import imageio
from flow_utils import bilinear_warp
import cv2

import psutil

config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
# ====================== HYPER-PARAMETERS ===========================###
# Adam
beta1 = 0.9
lr_decay = 0.1
vgg_conNum = 6

# evaluation
SIZE_SUBBLOCK = 168
SIZE_OVERLAP = 4

calculateOP = False # by Zihao

def inference(b_imgs0, b_imgs1, model):
    b,h,w,c = b_imgs0.shape
    flows = []
    for i in range(b):
        img0 = (b_imgs0[i,:,:,0]*255).astype(np.uint8)
        img1 = (b_imgs1[i,:,:,0]*255).astype(np.uint8)
        flow = model.calc(img0, img1, None)
        flows.append(flow)
    return np.array(flows)


def train():
    ###====================== READ INPUT PARAMETERS ===========================###

    train_hr_path = tl.global_flag['trainHR']
    train_lr_path = tl.global_flag['trainLR']
    valid_blocks_path = tl.global_flag['validBlocks']
    createValid = int(tl.global_flag['createValid'])
    ratio = float(tl.global_flag['ratio'])
    if ratio == 1:
        # batch_size = 64
        # generator = SRCNN_g
        # generator_str = 'SRCNN_g'
        # valid_block_size = 192
        batch_size = 12 # hardcoded by Zihao, used to be 16
        generator = ESRGAN_OutCascading_InShareSkip_Rethinking_efficient_g
        generator_str = 'ESRGAN_OutCascading_InShareSkip_Rethinking_efficient_g'
        valid_block_size = 96
    else:
        batch_size = 12 # hardcoded by Zihao, used to be 16
        generator = SRGAN_g_new
        generator_str = 'SRGAN_g_new'
        valid_block_size = 96
    trainName = tl.global_flag['trainName']
    subName = tl.global_flag['subName']
    results_path = tl.global_flag['resultsFolder']
    block_size = int(tl.global_flag['blockSize'])
    inputFormat = tl.global_flag['inputFormat']
    rotate_flag = int(tl.global_flag['rotate'])
    n_layers = int(tl.global_flag['nlayers'])
    decay_every = int(tl.global_flag['decayEpochs'])
    lr_init = float(tl.global_flag['paramLR_INIT'])
    lr_gan = float(tl.global_flag['paramLR_GAN'])
    lr_decay = float(tl.global_flag['paramLD'])
    n_epoch_init = int(tl.global_flag['nepochs'])
    BNflag = int(tl.global_flag['BN'])
    if inputFormat == "Y":
        input_depth = 1
    elif inputFormat.strip() == "RGB":
        input_depth = 3
    else:
        raise Exception("Input format should be either Y or RGB.")

    if BNflag == 0:
        BNflag = False
    else:
        BNflag = True
    lossFun = tl.global_flag['loss']
    isGAN = int(tl.global_flag['GAN'])
    if isGAN == 1:
        n_epoch = int(tl.global_flag['nepochs_GAN'])
        decay_every_gan = int(tl.global_flag['decayEpochsGAN'])
    else:
        n_epoch = 0
    readBatch_flag = int(tl.global_flag['readBatch_flag'])

    interpUV = tl.global_flag['interpUV']

    results_folder = os.path.join(results_path, trainName)
    sub_folder = os.path.join(results_folder, subName)
    logFilename = os.path.join(sub_folder, 'logFile.txt')
    lossInitFilename = os.path.join(sub_folder, 'loss_init.csv')
    lossFilename = os.path.join(sub_folder, 'loss.csv')

    # ni = int(np.sqrt(batch_size)) # by Zihao, no longer used when batch_size != 16
    for n_row in range(int(np.sqrt(batch_size)),1,-1):
        if batch_size%n_row == 0:
            break
    ni = [n_row,batch_size//n_row]


    # create folders to save result images and trained model
    tl.files.exists_or_mkdir(results_folder)
    tl.files.exists_or_mkdir(results_path)
    save_dir_ginit = os.path.join(sub_folder, 'samples_ginit')
    save_dir_gan = os.path.join(sub_folder, 'samples_g')
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = os.path.join(sub_folder, 'models')
    tl.files.exists_or_mkdir(checkpoint_dir)

    g_init_model = os.path.join(checkpoint_dir, 'g_init_0.npz')  # epoch 0 means the model it starts learning from
    g_model = os.path.join(checkpoint_dir, 'g_0.npz')  # epoch 0 means the model it starts learning from
    d_model = os.path.join(checkpoint_dir, 'd_0.npz')  # epoch 0 means the model it starts learning from

    train_folder = os.path.join(results_folder, 'train_dataset')
    train_hr_img_path = os.path.join(train_folder, 'HR')
    train_lr_img_path = os.path.join(train_folder, 'LR')
    valid_hr_img_path = os.path.join(valid_blocks_path, 'HR')
    valid_lr_img_path = os.path.join(valid_blocks_path, 'LR')

    save_dir_OFwarp = os.path.join(sub_folder, 'temp_OFwarp')
    tl.files.exists_or_mkdir(save_dir_OFwarp)

    # prepare log files and print current input parameters

    if ratio == 1:
        log = "epoch,train_loss,train_mse,valid_mse,LR_mse\n"
    else:
        log = "epoch,train_losstrain_mse,valid_mse_Y,valid_mse_U,valid_mse_V\n"
    print2logFile(lossInitFilename, log)
    log = "epoch,d_loss,g_loss,pixel_loss,vgg,adv\n"
    #log = "epoch,d_loss,g_loss,pixel_loss,adv\n"
    print2logFile(lossFilename, log)

    log = "####Training compressed CNN####\n\nParameters:\n\ntrainHR: %s\ntrainLR: %s\nvalidation blocks: %s\n" \
          "ratio: %d\nblock size: %d\nbatch size: %d\nlearning rate init: %.2e\nlearning rate gan: %.2e\nbeta: %.3f\ninit epochs: %d\nlearning decay: %.2e every %d epochs\n" \
          "number of layers: %d\nRotate flag: %d\nBN = %r\nLoss = %s\nIs GAN = %d\ninterpUV = %s\n" % (
          train_hr_path, train_lr_path, valid_blocks_path, ratio, block_size, batch_size, lr_init, lr_gan, beta1,
          n_epoch_init, lr_decay, decay_every, n_layers, rotate_flag, BNflag, lossFun, isGAN, interpUV)
    if isGAN == 1:
        log = log + "GAN epochs: %d\nGAN decay: %.2e every %d epochs\n" % (n_epoch, lr_decay, decay_every_gan)

    print(log)
    print2logFile(logFilename, log)

    ###====================== CREATE TRAINING DATASET ===========================###
    train_hr_list = sorted(tl.files.load_file_list(path=train_hr_path, regx='.*.yuv', printable=False),
                           key=str.lower)  # names of the images
    train_lr_list = sorted(tl.files.load_file_list(path=train_lr_path, regx='.*.yuv', printable=False), key=str.lower)

    check_HR_LR_match(train_hr_list, train_lr_list)

    log = "train HR and LR datasets contain %d sequences\n" % (len(train_hr_list))
    print(log)
    print2logFile(logFilename, log)

    # train_hr_list = [item for item in train_hr_list for i in range(2)]

    ret_create = create_training_images(train_hr_list, train_hr_path, train_lr_list, train_lr_path, ratio, block_size,
                                        batch_size, rotate_flag, train_folder, 1, readBatch_flag, createValid)
    if ret_create == -1:
        log = "train_dataset folder already exists, will skip the creation of training samples\n"
        print(log)
        print2logFile(logFilename, log)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    # create placeholder variables which serve as the inputs to graphs

    with tf.device('/device:GPU:0 '):

        t_image = tf.placeholder('float32', [batch_size, block_size, block_size, 9],
                                 name='t_image')  # input LR image

        im_size_HR = int(block_size * ratio)

        t_image_up = tf.placeholder('float32', [batch_size, im_size_HR, im_size_HR, input_depth],
                                    name='t_target_image_up')

        t_target_image = tf.placeholder('float32', [batch_size, im_size_HR, im_size_HR, input_depth],
                                        name='t_target_image')  # HR image (target image)

        t_image_valid = tf.placeholder('float32', [batch_size, valid_block_size, valid_block_size, 9],
                                       name='t_image_valid')  # input LR image

        t_image_valid_up = tf.placeholder('float32',
                                          [batch_size, int(ratio * valid_block_size), int(ratio * valid_block_size),
                                           input_depth], name='t_image_valid_up')  # input LR image

        net_g = generator(t_image, t_image_up, is_train=True, input_depth=input_depth, n_layers=n_layers, BN=BNflag,
                          ratio=ratio, reuse=False)
        net_g.print_params(False)

        net_g_test = generator(t_image_valid, t_image_valid_up, is_train=False, input_depth=input_depth,
                               n_layers=n_layers, BN=BNflag, ratio=ratio, reuse=True)

        # ###========================== DEFINE TRAIN OPS ==========================###

        # Generative network loss function
        if lossFun == "L2":
            pixel_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)  # L2 loss
        elif lossFun == "L1":
            if input_depth == 3:
                pixel_loss_Y = tl.cost.absolute_difference_error(net_g.outputs[:, :, :, 0], t_target_image[:, :, :, 0],
                                                                 is_mean=True)  # L1 loss Y
                pixel_loss_U = tl.cost.absolute_difference_error(net_g.outputs[:, 0::2, 0::2, 1],
                                                                 t_target_image[:, 0::2, 0::2, 1],
                                                                 is_mean=True)  # L1 loss U
                pixel_loss_V = tl.cost.absolute_difference_error(net_g.outputs[:, 0::2, 0::2, 2],
                                                                 t_target_image[:, 0::2, 0::2, 2],
                                                                 is_mean=True)  # L1 loss V

                pixel_loss = 4 * pixel_loss_Y + pixel_loss_U + pixel_loss_V
            else:
                pixel_loss = tl.cost.absolute_difference_error(net_g.outputs, t_target_image, is_mean=True)  # L1 loss Y
        elif lossFun == "Lap+L1":
            if input_depth == 3:
                lap_loss_Y = laploss(net_g.outputs[:, :, :, 0:1], t_target_image[:, :, :, 0:1])              # loss Y
                lap_loss_U = laploss(net_g.outputs[:, 0::2, 0::2, 1:2], t_target_image[:, 0::2, 0::2, 1:2], max_levels=3)  # loss U
                lap_loss_V = laploss(net_g.outputs[:, 0::2, 0::2, 2:3], t_target_image[:, 0::2, 0::2, 2:3], max_levels=3)  # loss V
                lap_loss = 4 * lap_loss_Y + lap_loss_U + lap_loss_V

                l1_loss_Y = tl.cost.absolute_difference_error(net_g.outputs[:, :, :, 0], t_target_image[:, :, :, 0], is_mean=True)  # L1 loss Y
                l1_loss_U = tl.cost.absolute_difference_error(net_g.outputs[:, 0::2, 0::2, 1], t_target_image[:, 0::2, 0::2, 1], is_mean=True)  # L1 loss U
                l1_loss_V = tl.cost.absolute_difference_error(net_g.outputs[:, 0::2, 0::2, 2], t_target_image[:, 0::2, 0::2, 2], is_mean=True)  # L1 loss V
                l1_loss = 4 * l1_loss_Y + l1_loss_U + l1_loss_V                
            else:
                l1_loss = tl.cost.absolute_difference_error(net_g.outputs, t_target_image, is_mean=True)  # L1 loss Y
                lap_loss = laploss(net_g.outputs, t_target_image)  # L1 loss Y
            pixel_loss = 10.*lap_loss + 1.*l1_loss
        elif lossFun == "SSIM":
            pixel_loss = 1 - tf.reduce_mean(tf.image.ssim(net_g.outputs, t_target_image, max_val=1.0))  # SSIM loss
        elif lossFun == "MSSSIM":
            pixel_loss = 1 - tf.reduce_mean(
                tf.image.ssim_multiscale(net_g.outputs, t_target_image, max_val=1.0))  # MS-SSIM loss
        else:
            raise Exception(
                "Error: the pixel loss argument (--loss) should be one of the following options: L1, L2, SSIM, MSSSIM.")


        g_vars = tl.layers.get_variables_with_name(generator_str, True, True)

        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr_init, trainable=False)

        ## Train generator network without GAN
        g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(pixel_loss, var_list=g_vars)


        ###========================== RESTORE MODEL =============================###
        sess = tf.Session(config=config)
        tl.layers.initialize_global_variables(sess)
        for i, p in enumerate(net_g.all_params):
            try:
                # print("  param {:3}: {:15} (mean: {:<18}, median: {:<18}, std: {:<18})   {}".format(i, str(p.eval().shape), p.eval().mean(), np.median(p.eval()), p.eval().std(), p.name))
                val = p.eval(session=sess)
                print(
                    "  param {:3}: {:20} {:15}    {} (mean: {:<18}, median: {:<18}, std: {:<18}, max: {:<18}, min: {:<18})   ".format(
                        i, p.name, str(val.shape), p.dtype.name, val.mean(), np.median(val), val.std(), val.max(),
                        val.min()))
            except Exception as e:
                print(str(e))
                raise Exception(
                    "Hint: print params details after tl.layers.initialize_global_variables(sess) or use network.print_params(False).")

        # load previously trained model as a starting point for training
        # In order to load a model, put this model in the model folder as g_init_0.npz
        tl.files.load_and_assign_npz(sess=sess, name=g_init_model, network=net_g)

        # Initialize flow estimator
        flow_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow_estimator.setFinestScale(0)
        print('Flow estimator initialised successfully!')


        ###============================= GET TRAINING AND VALIDATION IMAGES ===============================###

        if readBatch_flag == 1:
            train_hr_img_path = os.path.join(train_folder, "batch_blocks", "HR")
            train_lr_img_path = os.path.join(train_folder, "batch_blocks", "LR")

        train_hr_img_list = sorted(tl.files.load_file_list(path=train_hr_img_path, regx='.*.image', printable=False))  # names of the images
        train_lr_img0_list = sorted(tl.files.load_file_list(path=train_lr_img_path, regx='.*_0.image', printable=False))
        train_lr_img1_list = sorted(tl.files.load_file_list(path=train_lr_img_path, regx='.*_1.image', printable=False))
        train_lr_img2_list = sorted(tl.files.load_file_list(path=train_lr_img_path, regx='.*_2.image', printable=False))
        valid_hr_img_list = sorted(tl.files.load_file_list(path=valid_hr_img_path, regx='.*.image', printable=False))  # names of the images
        valid_lr_img0_list = sorted(tl.files.load_file_list(path=valid_lr_img_path, regx='.*_0.image', printable=False))
        valid_lr_img1_list = sorted(tl.files.load_file_list(path=valid_lr_img_path, regx='.*_1.image', printable=False))
        valid_lr_img2_list = sorted(tl.files.load_file_list(path=valid_lr_img_path, regx='.*_2.image', printable=False))

        assert (len(train_hr_img_list) == len(
            train_lr_img0_list)), "number of blocks used for training between HR and LR are not consistent"
        assert (len(valid_hr_img_list) == len(
            valid_lr_img0_list)), "number of blocks used for validation between HR and LR are not consistent"

        log = "total number of training blocks: %d\n" % (len(train_hr_img_list))
        print(log)
        print2logFile(logFilename, log)

        log = "total number of validation blocks: %d\n" % (len(valid_hr_img_list))
        print(log)
        print2logFile(logFilename, log)

        train_hr_img_list, train_lr_img0_list, train_lr_img1_list, train_lr_img2_list  = unison_shuffle(train_hr_img_list, train_lr_img0_list, train_lr_img1_list, train_lr_img2_list)  # fixed seed

        valid_hr_img_list, valid_lr_img0_list, valid_lr_img1_list, valid_lr_img2_list = unison_shuffle(valid_hr_img_list, valid_lr_img0_list, valid_lr_img1_list, valid_lr_img2_list)  # fixed seed

        ## use first `batch_size` of train set to have a quick test during training
        for ii in range(0,8): # by Zihao, to test repeat reading, when batch_size=8
            if readBatch_flag == 1:
                sample_imgs_HR = np.asarray(
                    read_blocks_batch(valid_hr_img_list, path=valid_hr_img_path, mode='HR', ratio=ratio,
                                      block_size=valid_block_size, dimensions=input_depth, batch_size=batch_size)) # by Zihao, valid_hr_img_list[0] -> valid_hr_img_list[1]
                sample_imgs0_LR = np.asarray(
                    read_blocks_batch(valid_lr_img0_list, path=valid_lr_img_path, mode='LR', ratio=ratio,
                                      block_size=valid_block_size, dimensions=input_depth, batch_size=batch_size))
                sample_imgs1_LR = np.asarray(
                    read_blocks_batch(valid_lr_img1_list, path=valid_lr_img_path, mode='LR', ratio=ratio,
                                      block_size=valid_block_size, dimensions=input_depth, batch_size=batch_size))
                sample_imgs2_LR = np.asarray(
                    read_blocks_batch(valid_lr_img2_list, path=valid_lr_img_path, mode='LR', ratio=ratio,
                                      block_size=valid_block_size, dimensions=input_depth, batch_size=batch_size))
            else:
                sample_imgs_HR = np.asarray(
                    read_blocks(valid_hr_img_list[0:batch_size], path=valid_hr_img_path, mode='HR', ratio=ratio,
                                block_size=valid_block_size, dimensions=input_depth))
                sample_imgs0_LR = np.asarray(
                    read_blocks(valid_lr_img0_list[0:batch_size], path=valid_lr_img_path, mode='LR', ratio=ratio,
                                block_size=valid_block_size, dimensions=input_depth))
                sample_imgs1_LR = np.asarray(
                    read_blocks(valid_lr_img1_list[0:batch_size], path=valid_lr_img_path, mode='LR', ratio=ratio,
                                block_size=valid_block_size, dimensions=input_depth))
                sample_imgs2_LR = np.asarray(
                    read_blocks(valid_lr_img2_list[0:batch_size], path=valid_lr_img_path, mode='LR', ratio=ratio,
                                block_size=valid_block_size, dimensions=input_depth))
    
            print(sample_imgs_HR.shape)
            print(sample_imgs1_LR.shape)
    
            tl.vis.save_images(np.uint8(255 * yuv2rgb_multiple(sample_imgs1_LR, 1)), ni,
                               os.path.join(save_dir_ginit, 'train_sample_LR_{}.png'.format(ii)))
            tl.vis.save_images(np.uint8(255 * yuv2rgb_multiple(sample_imgs_HR, 1)), ni,
                               os.path.join(save_dir_ginit, 'train_sample_HR_{}.png'.format(ii)))
            sample_imgs_LR_up = resize_multiple(sample_imgs1_LR,
                                                size=[int(ratio * valid_block_size), int(ratio * valid_block_size)],
                                                interpUV=interpUV, format="YUV420")
            tl.vis.save_images(np.uint8(255 * yuv2rgb_multiple(sample_imgs_LR_up, 1)), ni,
                               os.path.join(save_dir_ginit, 'train_sample_HR_lanczos_{}.png'.format(ii)))
            if isGAN == 1:
                tl.vis.save_images(np.uint8(255 * yuv2rgb_multiple(sample_imgs1_LR, 1)), ni,
                                   os.path.join(save_dir_gan, 'train_sample_LR.png'))
                tl.vis.save_images(np.uint8(255 * yuv2rgb_multiple(sample_imgs_HR, 1)), ni,
                                   os.path.join(save_dir_gan, 'train_sample_HR.png'))
                tl.vis.save_images(np.uint8(255 * yuv2rgb_multiple(sample_imgs_LR_up, 1)), ni,
                                   os.path.join(save_dir_gan, 'train_sample_HR_lanczos.png'))

        read_blocks_batch([], path='', mode='', ratio=ratio, block_size=0, dimensions=0, batch_size=0) # by Zihao. Call this to reset some global variables
        
        if input_depth == 1:
            sample_imgs1_LR = np.expand_dims(sample_imgs1_LR, axis=3)

        ###============================= CALCULATE MEAN IMAGE ===============================###

        if ratio == 1:
            mean_dims_HR = np.array([0.0, 0.0, 0.0])
            mean_dims_LR = np.array([0.0, 0.0, 0.0])
            mean_value = 0.0
        else:
            mean_dims_HR = np.array([1, 1, 1])
            mean_dims_LR = np.array([1, 1, 1])
            mean_value = 1

        if input_depth == 1:
            mean_image_LR = mean_dims_LR[0] * np.ones((block_size, block_size, 1))
            mean_image_HR = mean_dims_HR[0] * np.ones((im_size_HR, im_size_HR, 1))
        else:
            mean_image_Y = mean_dims_LR[0] * np.ones((block_size, block_size, 1))
            mean_image_U = mean_dims_LR[1] * np.ones((block_size, block_size, 1))
            mean_image_V = mean_dims_LR[2] * np.ones((block_size, block_size, 1))
            mean_image_LR = np.concatenate((mean_image_Y, mean_image_U, mean_image_V), axis=2)

            mean_image_Y = mean_dims_HR[0] * np.ones((im_size_HR, im_size_HR, 1))
            mean_image_U = mean_dims_HR[1] * np.ones((im_size_HR, im_size_HR, 1))
            mean_image_V = mean_dims_HR[2] * np.ones((im_size_HR, im_size_HR, 1))
            mean_image_HR = np.concatenate((mean_image_Y, mean_image_U, mean_image_V), axis=2)

        # get mean block - batch
        mean_block_HR_batch = np.repeat(np.expand_dims(mean_image_HR, axis=0), batch_size, axis=0)
        mean_block_LR_batch = np.repeat(np.expand_dims(mean_image_LR, axis=0), batch_size, axis=0)

        if input_depth == 1:
            sample_imgs_LR_up = np.expand_dims(sample_imgs_LR_up, axis=3)

        if ratio != 1:
            sample_imgs1_LR = 2 * sample_imgs1_LR - mean_value
            sample_imgs_LR_up = 2 * sample_imgs_LR_up - mean_value

        ###========================= TRAIN INITIAL GENERATOR ====================###
        if isGAN == 0:
            ## fixed learning rate
            total_mse_LR = 0
            total_mse_loss_valid_LR = 0
            total_mse_loss_valid_LR_Y = 0
            total_mse_loss_valid_LR_U = 0
            total_mse_loss_valid_LR_V = 0
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** fixed learning rate: %f (for init G)\n" % (lr_init)
            print2logFile(logFilename, log)

            for epoch in range(1, n_epoch_init + 1):
                #logfile = open('./test.out','a+') # by Zihao
                #logfile.write('---------------\n')
                #logfile.close()
                
                epoch_time = time.time()
                total_loss, total_mse_UP, n_iter = 0, 0, 0

                if epoch != 1 and ((epoch - 1) % decay_every == 0):
                    new_lr_decay = lr_decay ** (epoch // decay_every)
                    sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                    log = " ** new learning rate: %f\n" % (lr_init * new_lr_decay)
                    print(log)
                    print2logFile(logFilename, log)

                if readBatch_flag == 1:
                    #rangeIdx = range(0, len(train_hr_img_list)) # in python2.7, range should be list. will need to be changed in python3
                    #rangeIdx_valid = range(0, len(valid_hr_img_list))
                    rangeIdx = range(0, len(train_hr_img_list)*16//batch_size) # by Zihao, for batch_size=8
                    rangeIdx_valid = range(0, len(valid_hr_img_list)*16//batch_size)
                else:
                    rangeIdx = range(0, int(len(train_hr_img_list) / batch_size) * batch_size, batch_size)
                    rangeIdx_valid = range(0, int(len(valid_hr_img_list) / batch_size) * batch_size, batch_size)

                t1=t2=t3=t4=t5=0
                for idx in rangeIdx:
                    
                    if idx%1000==0:
                        logfile = open('./progress/{}.out'.format(subName),'a+') # by Zihao
                        logfile.write('training step {}/{} in epoch {} ...\n'.format(idx,len(rangeIdx),epoch))
                        log_memory = psutil.virtual_memory()
                        logfile.write('PID {} memory used: {} GB\n'.format(os.getpid(),psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024))
                        logfile.close()
                    
                    lastclock = step_time = time.time()

                    #logfile = open('./test.out','a+') # by Zihao
                    #log_memory = psutil.virtual_memory()
                    #logfile.write('PID {} memory used [READING BATCH]: {} GB\n'.format(os.getpid(),psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024))
                    #logfile.close()
                    if readBatch_flag == 1:
                        print(block_size)
                        print(batch_size)
                        b_imgs_HR = np.asarray(
                            read_blocks_batch(train_hr_img_list, path=train_hr_img_path, mode='HR', ratio=ratio,
                                              block_size=block_size, dimensions=input_depth, batch_size=batch_size))
                        b_imgs0_LR = np.asarray(
                            read_blocks_batch(train_lr_img0_list, path=train_lr_img_path, mode='LR', ratio=ratio,
                                              block_size=block_size, dimensions=input_depth, batch_size=batch_size))
                        b_imgs1_LR = np.asarray(
                            read_blocks_batch(train_lr_img1_list, path=train_lr_img_path, mode='LR', ratio=ratio,
                                              block_size=block_size, dimensions=input_depth, batch_size=batch_size))
                        b_imgs2_LR = np.asarray(
                            read_blocks_batch(train_lr_img2_list, path=train_lr_img_path, mode='LR', ratio=ratio,
                                              block_size=block_size, dimensions=input_depth, batch_size=batch_size))                      
                    else:
                        b_imgs_HR = np.asarray(
                            read_blocks(train_hr_img_list[idx: idx + batch_size], path=train_hr_img_path, mode='HR',
                                        ratio=ratio, block_size=block_size, n_threads=16, dimensions=input_depth))
                        b_imgs0_LR = np.asarray(
                            read_blocks(train_lr_img0_list[idx: idx + batch_size], path=train_lr_img_path, mode='LR',
                                        ratio=ratio, block_size=block_size, n_threads=16, dimensions=input_depth))
                        b_imgs1_LR = np.asarray(
                            read_blocks(train_lr_img1_list[idx: idx + batch_size], path=train_lr_img_path, mode='LR',
                                        ratio=ratio, block_size=block_size, n_threads=16, dimensions=input_depth))
                        b_imgs2_LR = np.asarray(
                            read_blocks(train_lr_img2_list[idx: idx + batch_size], path=train_lr_img_path, mode='LR',
                                        ratio=ratio, block_size=block_size, n_threads=16, dimensions=input_depth))

                    b_imgs0_LR_up = resize_multiple(b_imgs0_LR, size=[int(ratio * block_size), int(ratio * block_size)],
                                                   interpUV=interpUV, format="YUV420")
                    b_imgs1_LR_up = resize_multiple(b_imgs1_LR, size=[int(ratio * block_size), int(ratio * block_size)],
                                                   interpUV=interpUV, format="YUV420")
                    b_imgs2_LR_up = resize_multiple(b_imgs2_LR, size=[int(ratio * block_size), int(ratio * block_size)],
                                                   interpUV=interpUV, format="YUV420")

                    if input_depth == 1:
                        b_imgs_HR = np.expand_dims(b_imgs_HR, axis=3)
                        b_imgs0_LR = np.expand_dims(b_imgs0_LR, axis=3)
                        b_imgs1_LR = np.expand_dims(b_imgs1_LR, axis=3)
                        b_imgs2_LR = np.expand_dims(b_imgs2_LR, axis=3)
                        b_imgs0_LR_up = np.expand_dims(b_imgs0_LR_up, axis=3)
                        b_imgs1_LR_up = np.expand_dims(b_imgs1_LR_up, axis=3)
                        b_imgs2_LR_up = np.expand_dims(b_imgs2_LR_up, axis=3)

                    if ratio != 1:
                        b_imgs_HR = 2 * b_imgs_HR - mean_block_HR_batch
                        b_imgs0_LR = 2 * b_imgs0_LR - mean_block_LR_batch
                        b_imgs1_LR = 2 * b_imgs1_LR - mean_block_LR_batch
                        b_imgs2_LR = 2 * b_imgs2_LR - mean_block_LR_batch
                        b_imgs0_LR_up = 2 * b_imgs0_LR_up - mean_block_HR_batch
                        b_imgs1_LR_up = 2 * b_imgs1_LR_up - mean_block_HR_batch
                        b_imgs2_LR_up = 2 * b_imgs2_LR_up - mean_block_HR_batch

                    #logfile = open('./test.out','a+') # by Zihao
                    #log_memory = psutil.virtual_memory()
                    #logfile.write('PID {} memory used [OF WARP]: {} GB\n'.format(os.getpid(),psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024))
                    #logfile.close()

                    t1 += time.time() - lastclock
                    lastclock = time.time()
                    
                    print("time to read blocks: %4.4fs" % (time.time() - step_time))
                    ## get flows and warp
                    if epoch == 1 and calculateOP:
                        flows_10 = inference(b_imgs1_LR, b_imgs0_LR, model=flow_estimator) # B,H,W,2
                        flows_12 = inference(b_imgs1_LR, b_imgs2_LR, model=flow_estimator) # B,H,W,2 
                    
                    #logfile = open('./test.out','a+') # by Zihao
                    #log_memory = psutil.virtual_memory()
                    #logfile.write('PID {} memory used [AFTER OF]: {} GB\n'.format(os.getpid(),psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024))
                    #logfile.write('b_imgs0_LR type: {}\n'.format(type(b_imgs0_LR)))
                    #logfile.write('flows_10 type: {}\n'.format(type(flows_10)))
                    #logfile.close()
                    
                    t2 += time.time() - lastclock
                    lastclock = time.time()
                    
                    #temp1 = bilinear_warp(b_imgs0_LR, flows_10)
                    #b_imgs10 = temp1.eval(session=sess)
                    #temp2 = bilinear_warp(b_imgs2_LR, flows_12)
                    #b_imgs12 = temp2.eval(session=sess)
                    if epoch == 1 and calculateOP:
                        b_imgs10 = bilinear_warp(b_imgs0_LR, flows_10)
                        b_imgs12 = bilinear_warp(b_imgs2_LR, flows_12)
                        np.save(os.path.join(save_dir_OFwarp,'{}_10.npy'.format(idx)),b_imgs10)
                        np.save(os.path.join(save_dir_OFwarp,'{}_12.npy'.format(idx)),b_imgs12)
                    else:
                        b_imgs10 = np.load(os.path.join(save_dir_OFwarp,'{}_10.npy'.format(idx)))
                        b_imgs12 = np.load(os.path.join(save_dir_OFwarp,'{}_12.npy'.format(idx)))
                    
                    t3 += time.time() - lastclock
                    lastclock = time.time()
                    
                    #logfile = open('./test.out','a+') # by Zihao
                    #logfile.write('b_imgs10 type: {}\n'.format(type(b_imgs10)))
                    #logfile.write('b_imgs10 shape: {}\n'.format(b_imgs10.shape))
                    #logfile.write('b_imgs0_LR type: {}\n'.format(type(b_imgs0_LR)))
                    #logfile.write('flows_10 type: {}\n'.format(type(flows_10)))
                    #logfile.close()
                    b_imgs_LR = np.concatenate([b_imgs10, b_imgs1_LR, b_imgs12], axis=3)
                    
                    ## update G
                    #logfile = open('./test.out','a+') # by Zihao
                    #log_memory = psutil.virtual_memory()
                    #logfile.write('PID {} memory used [BEFORE SESS]: {} GB\n'.format(os.getpid(),psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024))
                    #logfile.close()
                    b_imgs_UP, errM, g_vars_training, _ = sess.run([net_g.outputs, pixel_loss, g_vars, g_optim_init],
                                                                   {t_image: b_imgs_LR, t_image_up: b_imgs1_LR_up,
                                                                    t_target_image: b_imgs_HR})
                    #logfile = open('./test.out','a+') # by Zihao
                    #log_memory = psutil.virtual_memory()
                    #logfile.write('PID {} memory used [AFTER SESS]: {} GB\n'.format(os.getpid(),psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024))
                    #logfile.close()
                    t4 += time.time() - lastclock
                    lastclock = time.time()
                    
                    print("Epoch [%2d/%2d] %4d time: %4.4fs, loss: %.8f " % (
                    epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
                    total_loss += errM
                    n_iter += 1

                    if ratio != 1:
                        b_imgs_HR = (b_imgs_HR + mean_block_HR_batch) / 2
                        b_imgs_LR = (b_imgs_LR + mean_block_LR_batch) / 2
                        b_imgs_UP = (b_imgs_UP + mean_block_HR_batch) / 2

                    b_imgs_UP[b_imgs_UP < 0] = 0
                    b_imgs_UP[b_imgs_UP > 1] = 1

                    b_imgs_UP = chromaSub_420_multiple(b_imgs_UP)

                    ###### for debugging
                    # for block_idx in range(0, batch_size):
                    #     print("block # " + str(block_idx))
                    #     print("range before GAN: max = " + str(b_imgs_LR[block_idx].max()) + " min = " + str(
                    #         b_imgs_LR[block_idx].min()))
                    #     print("range after GAN: max = " + str(b_imgs_UP[block_idx].max()) + " min = " + str(
                    #         b_imgs_UP[block_idx].min()))
                    #     imageio.imwrite(os.path.join(train_folder, str(block_idx) + 'block_LR.png'),
                    #                     np.uint8(255 * np.squeeze(b_imgs_LR[block_idx])))
                    #     imageio.imwrite(os.path.join(train_folder, str(block_idx) + 'block_HR.png'),
                    #                     np.uint8(255 * np.squeeze(b_imgs_HR[block_idx])))
                    #     imageio.imwrite(os.path.join(train_folder, str(block_idx) + 'block_CNN.png'),
                    #                     np.uint8(255 * np.squeeze(b_imgs_UP[block_idx])))
                    #    # a = tl.prepro.imresize(np.uint8(255*b_imgs_LR[block_idx]), size=[im_size_HR, im_size_HR], interp='lanczos')
                    #    # imageio.imwrite(os.path.join(train_folder, str(block_idx) + 'block_Lanczos3.png'),a)

                    # b_imgs_HR_uint8 = np.uint8(255 * np.squeeze(b_imgs_HR))
                    # b_imgs_UP_uint8 = np.uint8(255 * np.squeeze(b_imgs_UP))
                    # mse_UP_batch = calculate_mse(b_imgs_UP_uint8, b_imgs_HR_uint8)/(255 * 255)
                    # total_mse_UP += mse_UP_batch

                    mse_UP_batch = calculate_mse(np.squeeze(b_imgs_UP), np.squeeze(b_imgs_HR))
                    total_mse_UP += mse_UP_batch

                    if epoch == 1:
                        if ratio == 1:
                            mse_LR_batch = calculate_mse(b_imgs1_LR, b_imgs_HR)
                            # print("mse_UP = " + str(errM) + "; mse_LR = " + str(mse_LR_batch))
                            # else:
                            #     b_imgs_lanczos = tl.prepro.imresize_multi(b_imgs_LR, size=[im_size_HR, im_size_HR], interp='lanczos')
                            #     mse_LR_batch = calculate_mse(b_imgs_lanczos, b_imgs_HR_uint8)/(255*255)
                            total_mse_LR += mse_LR_batch
                    
                    t5 += time.time() - lastclock
                    lastclock = time.time()
                    if idx%1000==0:
                        total_time = lastclock - epoch_time
                        logfile = open('./progress/{}.out'.format(subName),'a+') # by Zihao
                        logfile.write('total time consumed: {} s\n'.format(int(total_time)))
                        logfile.write('Reading blocks: {} %\n'.format(100*t1//total_time))
                        logfile.write('Optical flow: {} %\n'.format(100*t2//total_time))
                        logfile.write('Warpping: {} %\n'.format(100*t3//total_time))
                        logfile.write('Session running: {} %\n'.format(100*t4//total_time))
                        logfile.write('Post calculating: {} %\n'.format(100*t5//total_time))
                        logfile.close()
                    

                read_blocks_batch([], path='', mode='', ratio=ratio, block_size=0, dimensions=0, batch_size=0) # by Zihao. Call this to reset some global variables
                ##################################################################
                ### evaluate the model at each epoch on the validation dataset ###
                ##################################################################
                total_mse_loss_valid_Y, total_mse_loss_valid_U, total_mse_loss_valid_V = 0, 0, 0

                start_time = time.time()

                n_iter_valid = 0
                for idx in rangeIdx_valid:
                    if n_iter_valid%100==0:
                        logfile = open('./progress/{}.out'.format(subName),'a+') # by Zihao
                        logfile.write('validating step {}/{} in epoch {} ...\n'.format(n_iter_valid,len(rangeIdx_valid),epoch))
                        log_memory = psutil.virtual_memory()
                        logfile.write('PID {} memory used: {} GB\n'.format(os.getpid(),psutil.Process(os.getpid()).memory_info().rss/1024/1024/1024))
                        logfile.close()
                    
                    
                    if readBatch_flag == 1:
                        valid_imgs_HR = np.asarray(
                            read_blocks_batch(valid_hr_img_list, path=valid_hr_img_path, mode='HR', ratio=ratio,
                                              block_size=valid_block_size, dimensions=input_depth, batch_size=batch_size))
                        valid_imgs0_LR = np.asarray(
                            read_blocks_batch(valid_lr_img0_list, path=valid_lr_img_path, mode='LR', ratio=ratio,
                                              block_size=valid_block_size, dimensions=input_depth, batch_size=batch_size))
                        valid_imgs1_LR = np.asarray(
                            read_blocks_batch(valid_lr_img1_list, path=valid_lr_img_path, mode='LR', ratio=ratio,
                                              block_size=valid_block_size, dimensions=input_depth, batch_size=batch_size)) 
                        valid_imgs2_LR = np.asarray(
                            read_blocks_batch(valid_lr_img2_list, path=valid_lr_img_path, mode='LR', ratio=ratio,
                                              block_size=valid_block_size, dimensions=input_depth, batch_size=batch_size))
                    else:
                        valid_imgs_HR = np.asarray(
                            read_blocks(valid_hr_img_list[idx: idx + batch_size], path=valid_hr_img_path, mode='HR',
                                        ratio=ratio, block_size=valid_block_size, n_threads=16, dimensions=input_depth))
                        valid_imgs0_LR = np.asarray(
                            read_blocks(valid_lr_img0_list[idx: idx + batch_size], path=valid_lr_img_path, mode='LR',
                                        ratio=ratio, block_size=valid_block_size, n_threads=16, dimensions=input_depth))
                        valid_imgs1_LR = np.asarray(
                            read_blocks(valid_lr_img1_list[idx: idx + batch_size], path=valid_lr_img_path, mode='LR',
                                        ratio=ratio, block_size=valid_block_size, n_threads=16, dimensions=input_depth))
                        valid_imgs2_LR = np.asarray(
                            read_blocks(valid_lr_img2_list[idx: idx + batch_size], path=valid_lr_img_path, mode='LR',
                                        ratio=ratio, block_size=valid_block_size, n_threads=16, dimensions=input_depth))

                    valid_imgs0_LR_up = resize_multiple(valid_imgs0_LR,
                                                       size=[int(ratio * valid_block_size), int(ratio * valid_block_size)],
                                                       interpUV=interpUV, format="YUV420")
                    valid_imgs1_LR_up = resize_multiple(valid_imgs1_LR,
                                                       size=[int(ratio * valid_block_size), int(ratio * valid_block_size)],
                                                       interpUV=interpUV, format="YUV420")
                    valid_imgs2_LR_up = resize_multiple(valid_imgs2_LR,
                                                       size=[int(ratio * valid_block_size), int(ratio * valid_block_size)],
                                                       interpUV=interpUV, format="YUV420")

                    if input_depth == 1:
                        valid_imgs_HR = np.expand_dims(valid_imgs_HR, axis=3)
                        valid_imgs0_LR = np.expand_dims(valid_imgs0_LR, axis=3)
                        valid_imgs1_LR = np.expand_dims(valid_imgs1_LR, axis=3)
                        valid_imgs2_LR = np.expand_dims(valid_imgs2_LR, axis=3)
                        valid_imgs0_LR_up = np.expand_dims(valid_imgs0_LR_up, axis=3)
                        valid_imgs1_LR_up = np.expand_dims(valid_imgs1_LR_up, axis=3)
                        valid_imgs2_LR_up = np.expand_dims(valid_imgs2_LR_up, axis=3)

                    if ratio != 1:
                        valid_imgs0_LR = 2 * valid_imgs0_LR - mean_value
                        valid_imgs1_LR = 2 * valid_imgs1_LR - mean_value
                        valid_imgs2_LR = 2 * valid_imgs2_LR - mean_value
                        valid_imgs0_LR_up = 2 * valid_imgs0_LR_up - mean_value
                        valid_imgs1_LR_up = 2 * valid_imgs1_LR_up - mean_value
                        valid_imgs2_LR_up = 2 * valid_imgs2_LR_up - mean_value

                    ## get flows and warp
                    flows_10 = inference(valid_imgs1_LR, valid_imgs0_LR, model=flow_estimator) # B,H,W,2
                    flows_12 = inference(valid_imgs1_LR, valid_imgs2_LR, model=flow_estimator) # B,H,W,2
                    
                    #b_imgs10 = bilinear_warp(valid_imgs0_LR, flows_10).eval(session=sess)
                    #b_imgs12 = bilinear_warp(valid_imgs2_LR, flows_12).eval(session=sess)
                    b_imgs10 = bilinear_warp(valid_imgs0_LR, flows_10) # edited by Zihao
                    b_imgs12 = bilinear_warp(valid_imgs2_LR, flows_12)
                    
                    valid_imgs_LR = np.concatenate([b_imgs10, valid_imgs1_LR, b_imgs12], axis=3)

                    ## deploy G on validation images
                    valid_imgs_UP = sess.run(net_g_test.outputs,
                                             {t_image_valid: valid_imgs_LR, t_image_valid_up: valid_imgs1_LR_up})
                    logfile = open('./test.out','a+') # by Zihao
                    logfile.write('validated step {}/{} in epoch {} ...\n'.format(n_iter_valid,len(rangeIdx_valid),epoch))
                    logfile.close()
                    
                    if ratio != 1:
                        valid_imgs_UP = (valid_imgs_UP + mean_value) / 2

                    valid_imgs_UP[valid_imgs_UP < 0] = 0
                    valid_imgs_UP[valid_imgs_UP > 1] = 1

                    valid_imgs_UP = chromaSub_420_multiple(valid_imgs_UP)

                    # mse_valid_batch = calculate_mse(np.squeeze(valid_imgs_UP), np.squeeze(valid_imgs_HR))
                    # total_mse_loss_valid += mse_valid_batch

                    mse_valid_batch_Y = calculate_mse(np.squeeze(valid_imgs_UP[:, :, :, 0]),
                                                      np.squeeze(valid_imgs_HR[:, :, :, 0]))
                    mse_valid_batch_U = calculate_mse(np.squeeze(valid_imgs_UP[:, :, :, 1]),
                                                      np.squeeze(valid_imgs_HR[:, :, :, 1]))
                    mse_valid_batch_V = calculate_mse(np.squeeze(valid_imgs_UP[:, :, :, 2]),
                                                      np.squeeze(valid_imgs_HR[:, :, :, 2]))

                    total_mse_loss_valid_Y += mse_valid_batch_Y
                    total_mse_loss_valid_U += mse_valid_batch_U
                    total_mse_loss_valid_V += mse_valid_batch_V

                    if epoch == 1 and ratio == 1:
                        mse_valid_LR_Y = calculate_mse(np.squeeze(valid_imgs1_LR[:,:,:,0]), np.squeeze(valid_imgs_HR[:,:,:,0]))
                        mse_valid_LR_U = calculate_mse(np.squeeze(valid_imgs1_LR[:,:,:,1]), np.squeeze(valid_imgs_HR[:,:,:,1]))
                        mse_valid_LR_V = calculate_mse(np.squeeze(valid_imgs1_LR[:,:,:,2]), np.squeeze(valid_imgs_HR[:,:,:,2]))
                        total_mse_loss_valid_LR_Y += mse_valid_LR_Y
                        total_mse_loss_valid_LR_U += mse_valid_LR_U
                        total_mse_loss_valid_LR_V += mse_valid_LR_V

                    n_iter_valid += 1

                    print("batch [%2d/%2d] loss: %.8f" % (idx, int(len(valid_hr_img_list)), mse_valid_batch_Y))

                read_blocks_batch([], path='', mode='', ratio=ratio, block_size=0, dimensions=0, batch_size=0) # by Zihao. Call this to reset some global variables
                
                print("[*] Average validation mse: %.8f, took %4.4fs" % (
                total_mse_loss_valid_Y / n_iter_valid, time.time() - start_time))

                ##### print to log file

                if ratio == 1:
                    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, loss_train: %.8f, mse_train: %.8f [PSNR = %.3f], mse_valid: %.8f [PSNR = %.3f], mse_train_LR: %.8f [PSNR = %.3f], mse_valid_LR: %.8f [PSNR = %.3f]\n" % \
                          (epoch, n_epoch_init, time.time() - epoch_time, total_loss / n_iter,
                           total_mse_UP / (n_iter), 10 * np.log10(1 / (total_mse_UP / (n_iter))),
                           total_mse_loss_valid_Y / n_iter_valid,
                           (6*(10 * np.log10(1 / (total_mse_loss_valid_Y / (n_iter_valid)))) + \
                               1*(10 * np.log10(1 / (total_mse_loss_valid_U / (n_iter_valid)))) + \
                                   1*(10 * np.log10(1 / (total_mse_loss_valid_V / (n_iter_valid)))))/8,
                           total_mse_LR / (n_iter), 10 * np.log10(1 / (total_mse_LR / (n_iter))),
                           total_mse_loss_valid_LR_Y / n_iter_valid,
                           (6*(10 * np.log10(1 / (total_mse_loss_valid_LR_Y / (n_iter_valid)))) + \
                               1*(10 * np.log10(1 / (total_mse_loss_valid_LR_U / (n_iter_valid)))) + \
                                   1*(10 * np.log10(1 / (total_mse_loss_valid_LR_V / (n_iter_valid)))))/8)
                    logStats = "%d,%.8f,%.8f,%.8f,%.8f\n" % (
                    epoch, total_loss / n_iter, total_mse_UP / (n_iter), total_mse_loss_valid_Y / n_iter_valid,
                    total_mse_LR / n_iter)
                else:
                    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, loss_train: %.8f, mse_train: %.8f [PSNR = %.3f], mse_valid_Y: %.8f [PSNR = %.3f], mse_valid_U: %.8f [PSNR = %.3f], mse_valid_V: %.8f [PSNR = %.3f]\n" % \
                          (epoch, n_epoch_init, time.time() - epoch_time,
                           total_loss / n_iter, total_mse_UP / (n_iter), 10 * np.log10(1 / (total_mse_UP / (n_iter))),
                           total_mse_loss_valid_Y / n_iter_valid,
                           10 * np.log10(1 / (total_mse_loss_valid_Y / (n_iter_valid))),
                           total_mse_loss_valid_U / n_iter_valid,
                           10 * np.log10(1 / (total_mse_loss_valid_U / (n_iter_valid))),
                           total_mse_loss_valid_V / n_iter_valid,
                           10 * np.log10(1 / (total_mse_loss_valid_V / (n_iter_valid))))
                    logStats = "%d,%.8f,%.8f,%.8f,%.8f,%.8f\n" % (
                    epoch, total_loss / n_iter, total_mse_UP / (n_iter), total_mse_loss_valid_Y / n_iter_valid,
                    total_mse_loss_valid_U / n_iter_valid, total_mse_loss_valid_V / n_iter_valid)

                print(log)
                print2logFile(logFilename, log)
                print2logFile(lossInitFilename, logStats)

                # ### save sample images

                # out = sess.run(net_g_test.outputs, {t_image_valid: valid_imgs_LR, t_image_valid_up: sample_imgs_LR_up})

                # if ratio != 1:
                #     out = (out + mean_value) / 2

                # out[out < 0] = 0
                # out[out > 1] = 1

                # out = chromaSub_420_multiple(out)

                # print("[*] save images")
                # tl.vis.save_images(np.uint8(255 * (yuv2rgb_multiple(out, 1))), [ni, ni],
                #                    os.path.join(save_dir_ginit, 'train_' + str(epoch) + '.png'))

                ### save model
                if (epoch != 1) and (epoch % 1 == 0):
                    tl.files.save_npz(net_g.all_params, name=os.path.join(checkpoint_dir, "g_init_" + str(epoch) + ".npz"),
                                      sess=sess)



def evaluate(afterTrain=False):
    valid_lr_img_path = tl.global_flag['evalLR']
    valid_hr_img_path = tl.global_flag['evalHR']
    testModel = tl.global_flag['testModel']
    # results_folder = tl.global_flag['resultsFolder']
    # trainName = tl.global_flag['trainName']
    # subName = tl.global_flag['subName']
    # block_size = int(tl.global_flag['blockSize'])
    inputFormat = tl.global_flag['inputFormat']
    # test_epoch = tl.global_flag['testEpoch']
    nframes = int(tl.global_flag['nframes'])
    eval_inputType = tl.global_flag['eval_inputType']
    n_layers = int(tl.global_flag['nlayers'])
    ratio = float(tl.global_flag['ratio'])
    interpUV = tl.global_flag['interpUV']

    if ratio == 1:
        # batch_size = 64
        # generator = SRCNN_g
        # CNN_name = "VDSR"
        batch_size = 12 # no use
        generator = ESRGAN_OutCascading_InShareSkip_Rethinking_efficient_g
        CNN_name = 'BD9bit_YUV'
        valid_block_size = 96
    else:
        batch_size = 12 # no use
        generator = SRGAN_g_new
        CNN_name = "NEWCNN_YUV"
        # generator = SRGAN_g
        # CNN_name = "SRResNet"
    BNflag = int(tl.global_flag['BN'])
    splitBlocks = int(tl.global_flag['splitBlocks'])
    if inputFormat == "Y":
        input_depth = 1
    elif inputFormat == "RGB":
        input_depth = 3
    else:
        raise Exception("Input format should be either Y or RGB.")

    if BNflag == 0:
        BNflag = False
    else:
        BNflag = True

    isGAN = int(tl.global_flag['GAN'])
    if isGAN == 1:
        CNN_name = "GAN"

    readBatch_flag = int(tl.global_flag['readBatch_flag'])

    ni = int(np.sqrt(batch_size)) # no use

    ## create folders to save result images
    save_dir = valid_hr_img_path
    # tl.files.exists_or_mkdir(save_dir)
    # eva_total_filename = os.path.join(save_dir, "evaluation.txt")
    # print2logFile(eva_total_filename, "MSE, PSNR\n")
    checkpoint_model = testModel

    ###====================== Validation dataset ===========================###
    if eval_inputType == "YUV":

        with tf.device("/gpu:0"):

            if os.path.isfile(valid_lr_img_path):
                valid_lr_img_list = [ntpath.basename(valid_lr_img_path)]
                # valid_hr_img_list = [ntpath.basename(valid_hr_img_path)]
                valid_lr_img_path = ntpath.dirname(valid_lr_img_path)
                valid_hr_img_path = ntpath.dirname(valid_hr_img_path)
                print(valid_lr_img_list)
                # print(valid_hr_img_list)
                print(valid_lr_img_path)
                # print(valid_hr_img_path)
            else:
                valid_lr_img_list = sorted(
                    tl.files.load_file_list(path=valid_lr_img_path, regx='.*.yuv', printable=False))
                # valid_hr_img_list = sorted(
                #    tl.files.load_file_list(path=valid_hr_img_path, regx='.*.yuv', printable=False))

            print("sequences to evaluate: " + str(valid_lr_img_list))

            # Restore mean blocks from training dataset
            # mean_block_HR, mean_block_LR, _ = restore_mean_block(save_folder=os.path.join(results_folder, trainName, 'train_dataset'), ratio=ratio, block_size=block_size)
            # mean_dims_HR = np.mean(np.mean(mean_block_HR,axis=0),axis=0)
            # mean_dims_LR = np.mean(np.mean(mean_block_LR,axis=0),axis=0)

            # override mean values by 0.5 which makes the input of the CNN range from [-0.5, 0,5]
            if ratio == 1:
                mean_dims_HR = np.array([0.0, 0.0, 0.0])
                mean_dims_LR = np.array([0.0, 0.0, 0.0])
            else:
                mean_value = 1
                # mean_dims_HR = np.array([mean_value, mean_value, mean_value])
                mean_dims_LR = np.array([mean_value, mean_value, mean_value])

            mse_valid_UP_total_Y, mse_valid_UP_total_U, mse_valid_UP_total_V = 0, 0, 0
            file_count = 0
            frame_count = 0
            for video_file in valid_lr_img_list:

                ###========================== DEFINE MODEL ============================###
                currVideo = videoParams()
                currVideo.filename = video_file
                print(video_file)
                currVideo.extractVideoParameters()
                currVideo.printParams()

                # currVideo_hr = videoParams()
                # currVideo_hr.filename = video_file_hr
                # currVideo_hr.extractVideoParameters()
                # currVideo_hr.printParams()
                maxValue = pow(2, currVideo.bitDepth) - 1
                # eva_seq_filename = os.path.join(save_dir, "evaluation_" + video_file[:-4] + "_" + CNN_name + ".txt")
                # print2logFile(eva_seq_filename, "frame, MSE, PSNR\n")

                iframe = 0

                valid_lr_img0 = np.squeeze(
                    loadYUVfile(os.path.join(valid_lr_img_path, currVideo.filename), currVideo.width, currVideo.height,
                                np.array([iframe]), currVideo.colorSampling, currVideo.bitDepth))
                valid_lr_img1 = np.squeeze(
                    loadYUVfile(os.path.join(valid_lr_img_path, currVideo.filename), currVideo.width, currVideo.height,
                                np.array([iframe]), currVideo.colorSampling, currVideo.bitDepth))
                valid_lr_img2 = np.squeeze(
                    loadYUVfile(os.path.join(valid_lr_img_path, currVideo.filename), currVideo.width, currVideo.height,
                                np.array([iframe+1]), currVideo.colorSampling, currVideo.bitDepth))
                # valid_hr_img = np.squeeze(
                #    loadYUVfile(os.path.join(valid_hr_img_path, currVideo_hr.filename), currVideo_hr.width,
                #                currVideo_hr.height,
                #                np.array([iframe]), currVideo_hr.colorSampling, currVideo_hr.bitDepth))

                # convert LR image to RGB if input depth is 3
                # if input_depth == 3:
                #    valid_lr_img = yuv2rgb(np.squeeze(valid_lr_img), currVideo_hr.bitDepth)

                size_LR = valid_lr_img1.shape
                size_HR = valid_lr_img1.shape

                if splitBlocks == 1:

                    SIZE_SUBBLOCK = math.gcd(size_LR[0], size_LR[1]) + SIZE_OVERLAP * 2

                    # if size_LR[0] == 1080:
                    #    SIZE_SUBBLOCK = 128
                    # else:
                    #    SIZE_SUBBLOCK = 128 #168

                    t_image = tf.placeholder('float32', [None, SIZE_SUBBLOCK, SIZE_SUBBLOCK, 9],
                                             name='input_image')  # the 1 in the last dimension is because we are just using the Y channel

                    t_image_up = tf.placeholder('float32',
                                                [None, int(SIZE_SUBBLOCK * ratio), int(SIZE_SUBBLOCK * ratio),
                                                 input_depth], name='input_image_up')

                else:
                    t_image = tf.placeholder('float32', [None, size_LR[0], size_LR[1], 9],
                                             name='input_image')  # the 1 in the last dimension is because we are just using the Y channel

                    t_image_up = tf.placeholder('float32', [None, size_HR[0], size_HR[1], input_depth],
                                                name='input_image_up')

                if file_count == 0 and afterTrain == False:
                    net_g = generator(t_image, t_image_up, is_train=False, input_depth=input_depth, n_layers=n_layers,
                                      BN=BNflag, ratio=ratio, reuse=False)
                else:
                    net_g = generator(t_image, t_image_up, is_train=False, input_depth=input_depth, n_layers=n_layers,
                                      BN=BNflag, ratio=ratio, reuse=True)

                ###========================== RESTORE G =============================###
                sess = tf.Session(config=config)
                tl.layers.initialize_global_variables(sess)

                if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_model,
                                                network=net_g) is False:
                    raise Exception("Error loading trained model.")
                
                # Initialize flow estimator
                flow_estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                flow_estimator.setFinestScale(0)
                print('Flow estimator initialised successfully!')

                ##========================== Get mean frame =========================###

                if input_depth == 1:
                    mean_image_LR = mean_dims_LR[0] * np.ones((size_LR[0], size_LR[1], 1))
                    mean_image_HR = mean_dims_HR[0] * np.ones((1, size_HR[0], size_HR[1], 1))
                else:
                    mean_image_Y = mean_dims_LR[0] * np.ones((size_LR[0], size_LR[1], 1))
                    mean_image_U = mean_dims_LR[1] * np.ones((size_LR[0], size_LR[1], 1))
                    mean_image_V = mean_dims_LR[2] * np.ones((size_LR[0], size_LR[1], 1))
                    mean_image_LR = np.concatenate((mean_image_Y, mean_image_U, mean_image_V), axis=2)

                    mean_image_Y = mean_dims_HR[0] * np.ones((size_HR[0], size_HR[1], 1))
                    mean_image_U = mean_dims_HR[1] * np.ones((size_HR[0], size_HR[1], 1))
                    mean_image_V = mean_dims_HR[2] * np.ones((size_HR[0], size_HR[1], 1))
                    mean_image_HR = np.concatenate((mean_image_Y, mean_image_U, mean_image_V), axis=2)

                ##======================= Iterate on every frame ====================###

                size_subblock_HR = np.zeros((2, 1))
                size_overlap_HR = np.zeros((2, 1))

                # split in blocks
                if splitBlocks == 1:
                    h_blocks = int(size_LR[0] / (SIZE_SUBBLOCK - SIZE_OVERLAP * 2))
                    w_blocks = int(size_LR[1] / (SIZE_SUBBLOCK - SIZE_OVERLAP * 2))
                    size_subblock_HR[0] = int(SIZE_SUBBLOCK * ratio)
                    size_subblock_HR[1] = int(SIZE_SUBBLOCK * ratio)
                    size_overlap_HR[0] = int(SIZE_OVERLAP * ratio)
                    size_overlap_HR[1] = int(SIZE_OVERLAP * ratio)

                    width_start = np.zeros(w_blocks, dtype=int)
                    width_end = np.zeros(w_blocks, dtype=int)
                    height_start = np.zeros(h_blocks, dtype=int)
                    height_end = np.zeros(h_blocks, dtype=int)

                    for i_w in range(0, w_blocks):
                        if i_w == 0:
                            width_start[i_w] = 0
                        elif i_w == 1 or i_w == w_blocks - 1:
                            width_start[i_w] = width_start[i_w - 1] + (SIZE_SUBBLOCK - SIZE_OVERLAP * 3)
                        else:
                            width_start[i_w] = width_start[i_w - 1] + (SIZE_SUBBLOCK - SIZE_OVERLAP * 2)
                        width_end[i_w] = width_start[i_w] + SIZE_SUBBLOCK

                    for i_h in range(0, h_blocks):
                        if i_h == 0:
                            height_start[i_h] = 0
                        elif i_h == 1 or i_h == h_blocks - 1:
                            height_start[i_h] = height_start[i_h - 1] + (SIZE_SUBBLOCK - SIZE_OVERLAP * 3)
                        else:
                            height_start[i_h] = height_start[i_h - 1] + (SIZE_SUBBLOCK - SIZE_OVERLAP * 2)
                        height_end[i_h] = height_start[i_h] + SIZE_SUBBLOCK
                else:
                    h_blocks = 1
                    w_blocks = 1
                    size_subblock_HR[0] = int(size_HR[0])
                    size_subblock_HR[1] = int(size_HR[1])
                    size_overlap_HR[0] = 0
                    size_overlap_HR[1] = 0

                    width_start = np.array([0], dtype=int)
                    width_end = np.array([size_HR[1]], dtype=int)
                    height_start = np.array([0], dtype=int)
                    height_end = np.array([size_HR[0]], dtype=int)

                mse_valid_UP_seq_Y, mse_valid_UP_seq_U, mse_valid_UP_seq_V = 0, 0, 0

                if nframes == 0:
                    statinfo = os.stat(os.path.join(valid_lr_img_path, currVideo.filename))
                    nframes = statinfo.st_size / math.ceil(currVideo.bitDepth / 8) / size_LR[0] / size_LR[
                        1] / 1.5  # for 420 only

                while valid_lr_img1.ndim != 0 and iframe < nframes:

                    print("[*] upsampling sequence " + video_file + " frame " + str(iframe + 1))
                    start_time = time.time()

                    if input_depth == 1:
                        valid_lr_img1_input = normalize_array(valid_lr_img1[:, :, 0],
                                                             currVideo.bitDepth)  # keep only the Y channel
                        valid_lr_img1_input = np.expand_dims(valid_lr_img1_input, axis=2)
                    else:
                        valid_lr_img0_input = normalize_array(valid_lr_img0, currVideo.bitDepth)
                        valid_lr_img1_input = normalize_array(valid_lr_img1, currVideo.bitDepth)
                        valid_lr_img2_input = normalize_array(valid_lr_img2, currVideo.bitDepth)

                    # subtract mean
                    if ratio != 1:
                        valid_lr_img1_input = valid_lr_img1_input * 2 - mean_image_LR

                    # print("range before GAN: max = " + str(valid_lr_img.max()) + " min = " + str(valid_lr_img.min()))
                    # scipy.misc.imsave(currVideo.seqName + '_beforeGAN.jpg', valid_lr_img)

                    genFrame = np.zeros((1, size_HR[0], size_HR[1], input_depth))
                    count_blocks = 0

                    t1=t2=t3=t4=0
                    print("number of blocks:")
                    print("w_blocks = {}".format(w_blocks))
                    print("h_blocks = {}".format(h_blocks))
                    for i_w in range(0, w_blocks):
                        for i_h in range(0, h_blocks):

                            valid_lr_img0_block = valid_lr_img0_input[height_start[i_h]:height_end[i_h],
                                                 width_start[i_w]:width_end[i_w], :]
                            valid_lr_img1_block = valid_lr_img1_input[height_start[i_h]:height_end[i_h],
                                                 width_start[i_w]:width_end[i_w], :]
                            valid_lr_img2_block = valid_lr_img2_input[height_start[i_h]:height_end[i_h],
                                                 width_start[i_w]:width_end[i_w], :]

                            size_valid_block = np.shape(valid_lr_img1_block)

                            valid_lr_img_block_up = resize_single(valid_lr_img1_block,
                                                                  size=[int(ratio * size_valid_block[0]),
                                                                        int(ratio * size_valid_block[1])],
                                                                  interpUV=interpUV, format="YUV420")

                            # valid_lr_img_block_up = resize(valid_lr_img_block, [int(ratio * size_valid_block[0]), int(ratio * size_valid_block[1])], order=3, mode='reflect')
                            # valid_lr_img_block_up = tl.prepro.imresize_multi(valid_lr_img_block, size=[int(ratio * size_valid_block[0]), int(ratio * size_valid_block[1])])

                            ###======================= EVALUATION =============================###
                            lastclock = blocktime = time.time()
                            
                            flows_10 = inference(valid_lr_img1_block[None,:,:,:], valid_lr_img0_block[None,:,:,:], model=flow_estimator) # B,H,W,2
                            flows_12 = inference(valid_lr_img1_block[None,:,:,:], valid_lr_img2_block[None,:,:,:], model=flow_estimator) # B,H,W,2
                            
                            t1 += time.time() - lastclock
                            lastclock = time.time()
                            
                            b_imgs10 = bilinear_warp(valid_lr_img0_block[None,:,:,:], flows_10)#.eval(session=sess) # by Zihao changed to numpy 
                            b_imgs12 = bilinear_warp(valid_lr_img2_block[None,:,:,:], flows_12)#.eval(session=sess)
                            
                            t2 += time.time() - lastclock
                            lastclock = time.time()
                            
                            valid_lr_img_block = np.concatenate([b_imgs10, valid_lr_img1_block[None,:,:,:], b_imgs12], axis=3)
                            genBlock = sess.run(net_g.outputs,
                                                {t_image: valid_lr_img_block, t_image_up: valid_lr_img_block_up[None,:,:,:]})

                            t3 += time.time() - lastclock
                            lastclock = time.time()
                            
                            width_start_genFrame = i_w * (size_subblock_HR[1] - size_overlap_HR[1] * 2)
                            width_end_genFrame = width_start_genFrame + size_subblock_HR[1] - size_overlap_HR[1] * 2

                            if i_w == 0:
                                width_start_genBlock = 0
                            elif i_w == w_blocks - 1:
                                width_start_genBlock = size_overlap_HR[1] * 2
                            else:
                                width_start_genBlock = size_overlap_HR[1]

                            width_end_genBlock = width_start_genBlock + size_subblock_HR[1] - size_overlap_HR[1] * 2

                            height_start_genFrame = i_h * (size_subblock_HR[0] - size_overlap_HR[0] * 2)
                            height_end_genFrame = height_start_genFrame + size_subblock_HR[0] - size_overlap_HR[0] * 2

                            if i_h == 0:
                                height_start_genBlock = 0
                            elif i_h == h_blocks - 1:
                                height_start_genBlock = size_overlap_HR[0] * 2
                            else:
                                height_start_genBlock = size_overlap_HR[0]

                            height_end_genBlock = height_start_genBlock + size_subblock_HR[0] - size_overlap_HR[0] * 2

                            genFrame[:, int(height_start_genFrame):int(height_end_genFrame),
                            int(width_start_genFrame):int(width_end_genFrame), :] = \
                                genBlock[:, int(height_start_genBlock):int(height_end_genBlock),
                                int(width_start_genBlock):int(width_end_genBlock), :]

                            count_blocks = count_blocks + 1

                            t4 += time.time() - lastclock
                            lastclock = time.time()
                            
                            total_time = lastclock - start_time
                            logfile = open('./CNN/progress/{}.out'.format(video_file.split('_')[0]),'a+') # by Zihao
                            logfile.write('total time consumed: {} s\n'.format(int(total_time)))
                            logfile.write('Optical flow: {} %\n'.format(100*t1//total_time))
                            logfile.write('Warpping: {} %\n'.format(100*t2//total_time))
                            logfile.write('Session running: {} %\n'.format(100*t3//total_time))
                            logfile.write('Post calculating: {} %\n'.format(100*t4//total_time))
                            logfile.write(''.join(os.popen('nvidia-smi').readlines()))
                            logfile.write('\n')
                            logfile.close()

                    # print("range after GAN: max = " + str(genFrame.max()) + " min = " + str(genFrame.min()))
                    # scipy.misc.imsave(os.path.join(save_dir,currVideo.seqName + '_afterGAN.png'), np.squeeze(genFrame))

                    print("[*] upsampling took: %4.4fs, LR size: %s /  generated HR size: %s" % (
                        time.time() - start_time, valid_lr_img1_input.shape, genFrame.shape))

                    if ratio != 1:
                        genFrame = (genFrame + mean_image_HR) / 2  # add mean that was previously subtracted

                    if iframe == 0:
                        mode = 'wb'
                    else:
                        mode = 'ab'

                    size_HR = genFrame[0].shape

                    genFrame = inverse_normalize_array(genFrame[0], currVideo.outBitDepth)

                    # if input_depth==1:
                    # if size_LR[0] == size_HR[0] and size_LR[1] == size_HR[1]:
                    #     genFrameYUV = np.stack((genFrame[:, :, 0], valid_lr_img[:, :, 1], valid_lr_img[:, :, 2]), axis=2)
                    # else:
                    # start_time = time.time()
                    # im_lanczos = imresize_L3(valid_lr_img, scale=2)
                    # print("calculating lanczos resampling took: %4.4fs" % (time.time() - start_time))
                    # genFrameYUV = np.stack((genFrame[:, :, 0], im_lanczos[:, :, 1], im_lanczos[:, :, 2]), axis=2)
                    # im_lanczos = np.resize(im_lanczos, (1, size_HR[0], size_HR[1], 3))
                    # saveYUVfile(convertInt32(im_lanczos, currVideo.bitDepth),
                    #            os.path.join(save_dir, video_file[0:-4] + "_lanczos3.yuv"), mode,
                    #            currVideo.colorSampling, currVideo.bitDepth)

                    # genFrameYUV = genFrame[:, :, 0]
                    # else:
                    if input_depth == 1 and ratio == 1:
                        genFrameYUV = np.stack((genFrame[:, :, 0], valid_lr_img1[:, :, 1], valid_lr_img1[:, :, 2]),
                                               axis=2)
                    else:
                        # if input_depth == 3:
                        #    genFrameYUV = rgb2yuv(genFrame, currVideo_hr.bitDepth)
                        # else:
                        genFrameYUV = genFrame

                    # mse_valid_UP_Y = calculate_mse(genFrameYUV[:, :, 0], valid_hr_img[:, :, 0])
                    # mse_valid_UP_U = calculate_mse(genFrameYUV[:, :, 1], valid_hr_img[:, :, 1])
                    # mse_valid_UP_V = calculate_mse(genFrameYUV[:, :, 2], valid_hr_img[:, :, 2])
                    # mse_valid_UP_seq_Y += mse_valid_UP_Y
                    # mse_valid_UP_seq_U += mse_valid_UP_U
                    # mse_valid_UP_seq_V += mse_valid_UP_V
                    # mse_valid_UP_total_Y += mse_valid_UP_Y
                    # mse_valid_UP_total_U += mse_valid_UP_U
                    # mse_valid_UP_total_V += mse_valid_UP_V

                    if genFrameYUV.shape[2] == 3:
                        genFrameYUV = np.resize(genFrameYUV, (1, size_HR[0], size_HR[1], 3))
                        colorSave = currVideo.colorSampling
                    else:
                        genFrameYUV = np.resize(genFrameYUV, (1, size_HR[0], size_HR[1], 1))
                        colorSave = 'Y'

                    # log = "%d, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f\n" % (
                    # iframe, mse_valid_UP_Y, 10 * np.log10(pow(maxValue, 2) / (mse_valid_UP_Y)),
                    # mse_valid_UP_U, 10 * np.log10(pow(maxValue, 2) / (mse_valid_UP_U)), mse_valid_UP_V,
                    # 10 * np.log10(pow(maxValue, 2) / (mse_valid_UP_V)))
                    # print2logFile(eva_seq_filename, log)

                    print("[*] saving frame")

                    saveYUVfile(convertInt32(genFrameYUV, currVideo.outBitDepth),
                                save_dir, mode, colorSave,
                                currVideo.outBitDepth)

                    iframe += 1
                    frame_count += 1

                    valid_lr_img0 = valid_lr_img1
                    valid_lr_img1 = valid_lr_img2
                    # Only read the (iframe+1)th frame if iframe is not the last frame, otherwise keep valid_lr_img2 unchanged
                    if iframe != nframes-1:
                        valid_lr_img2 = np.squeeze(
                            loadYUVfile(os.path.join(valid_lr_img_path, currVideo.filename), currVideo.width,
                                        currVideo.height,
                                        np.array([iframe+1]), currVideo.colorSampling, currVideo.bitDepth))

                file_count = file_count + 1
    else:
        raise Exception("Evaluation input type: eval_inputType should be either 'YUV' or 'rawblocks'.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')

    # For training only
    parser.add_argument('--trainHR', type=str, help='path for the train HR folder')
    parser.add_argument('--trainLR', type=str, help='path for the train LR folder')
    parser.add_argument('--validBlocks', type=str, help='path for the validation blocks to be evaluated every epoch')
    parser.add_argument('--trainName', type=str, help='name for the current training (as a label)')
    parser.add_argument('--subName', type=str, help='name for the current training parameters (as a label): runs with '
                                                    'the trainName and different subNames will share the same training data')
    parser.add_argument('--resultsFolder', type=str,
                        help='location of folder to store results from traning/evalutation')

    parser.add_argument('--createValid', type=str, default="0", help='Enable to create validation dataset')
    parser.add_argument('--blockSize', type=str, default="96", help='size of LR blocks to use for training')
    parser.add_argument('--BN', type=str, default="0", help='whether to use batch normalization or not')
    parser.add_argument('--nepochs', type=str, default="200", help='Number of epochs to train for')
    parser.add_argument('--paramLR_INIT', type=str, default="0.0001", help='Initial learning rate for g init')
    parser.add_argument('--paramLR_GAN', type=str, default="0.0001", help='Initial learning rate for gan')
    parser.add_argument('--paramLD', type=str, default="0.1", help='learning rate decay')
    parser.add_argument('--decayEpochs', type=str, default="100",
                        help='Number of epochs at which to decay the learning rate for the initial generator training')
    parser.add_argument('--decayEpochsGAN', type=str, default="100",
                        help='Number of epochs at which to decay the learning rate for the GAN training')
    parser.add_argument('--GAN', type=str, default="0", help='If training a GAN (1) or not (0)')
    parser.add_argument('--nepochs_GAN', type=str, default="1000", help='Number of epochs to train GAN for')
    parser.add_argument('--interpUV', type=str, default="nearest",
                        help='How to interpolate the chroma channels: bicubic or nearest')
    parser.add_argument('--loss', type=str, default="L1",
                        help='options: L1, L2, SSIM for training the inial generator network (with or without the GAN)')

    # For evaluation only
    parser.add_argument('--evalHR', type=str, help='path for the evaluation HR folder (output)')
    parser.add_argument('--evalLR', type=str, help='path for the evaluation LR folder (input)')
    parser.add_argument('--testModel', type=str, help='the model is going to be tested')
    # parser.add_argument('--testEpoch', type=str, help='epoch of the generative network to evaluate')
    parser.add_argument('--nframes', type=str, default="0",
                        help='number of frames of the sequences to evaluate, 0 means using all frames')
    parser.add_argument('--eval_inputType', type=str, default="YUV", help='frame format of the sequences to evaluate')
    parser.add_argument('--splitBlocks', type=str, default="1",
                        help='Whether to split the LR frame into blocks when deploying the model (due to memory issues). '
                             'Only works for eval_inputType = YUV.')

    # For both training and evaluation
    parser.add_argument('--ratio', type=str, default="2", help='downsampling ratio to train or evaluate the model')
    parser.add_argument('--inputFormat', type=str, default="Y",
                        help='If the CNN input should be the in Y channel only or in RGB format.')
    parser.add_argument('--nlayers', type=str, default="16", help='number of convolutional layers to use')
    parser.add_argument('--rotate', type=str, default="1",
                        help='use rotation as a form of data augmentation for the creation of the training dataset')

    parser.add_argument('--readBatch_flag', type=str, default="1",
                        help='if training files are stores in batch size files or not')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['trainHR'] = args.trainHR
    tl.global_flag['trainLR'] = args.trainLR
    tl.global_flag['evalHR'] = args.evalHR
    tl.global_flag['evalLR'] = args.evalLR
    tl.global_flag['validBlocks'] = args.validBlocks
    tl.global_flag['ratio'] = args.ratio
    tl.global_flag['trainName'] = args.trainName
    tl.global_flag['subName'] = args.subName
    tl.global_flag['resultsFolder'] = args.resultsFolder
    tl.global_flag['blockSize'] = args.blockSize
    tl.global_flag['rotate'] = args.rotate
    tl.global_flag['createValid'] = args.createValid

    # training hiperparameters
    tl.global_flag['nlayers'] = args.nlayers
    tl.global_flag['paramLR_INIT'] = args.paramLR_INIT
    tl.global_flag['paramLR_GAN'] = args.paramLR_GAN
    tl.global_flag['paramLD'] = args.paramLD
    tl.global_flag['inputFormat'] = args.inputFormat
    tl.global_flag['nepochs'] = args.nepochs
    tl.global_flag['decayEpochs'] = args.decayEpochs
    tl.global_flag['decayEpochsGAN'] = args.decayEpochsGAN
    tl.global_flag['GAN'] = args.GAN
    tl.global_flag['nepochs_GAN'] = args.nepochs_GAN
    tl.global_flag['nepochs_GAN'] = args.nepochs_GAN
    tl.global_flag['BN'] = args.BN
    tl.global_flag['loss'] = args.loss
    tl.global_flag['interpUV'] = args.interpUV

    # evaluation
    # tl.global_flag['testEpoch'] = args.testEpoch
    tl.global_flag['testModel'] = args.testModel
    tl.global_flag['nframes'] = args.nframes
    tl.global_flag['readBatch_flag'] = args.readBatch_flag
    tl.global_flag['eval_inputType'] = args.eval_inputType
    tl.global_flag['splitBlocks'] = args.splitBlocks

    print("Input parameters")
    print("Mode = %s" % (tl.global_flag['mode']))
    print("Train HR folder = %s" % (tl.global_flag['trainHR']))
    print("Train LR folder = %s" % (tl.global_flag['trainLR']))
    print("Train label = %s" % (tl.global_flag['trainName']))
    print("Sub label = %s" % (tl.global_flag['subName']))
    print("Evaluation HR folder = %s" % (tl.global_flag['evalHR']))
    print("Evaluation LR folder = %s" % (tl.global_flag['evalLR']))
    print("Validation blocks path = %s" % (tl.global_flag['validBlocks']))
    print("Upsampling ratio = %s" % (tl.global_flag['ratio']))
    print("Results folder = %s" % (tl.global_flag['resultsFolder']))
    print("Block size = %s" % (tl.global_flag['blockSize']))
    print("Input format = %s" % (tl.global_flag['inputFormat']))
    print("Rotate flag = %s (0 = no rotation, 1 = rotation)" % (tl.global_flag['rotate']))
    print("Number of conv layers = %s" % (tl.global_flag['nlayers']))
    print("Number of epochs for training = %s" % (tl.global_flag['nepochs']))
    print("Learning rate init = %s" % (tl.global_flag['paramLR_INIT']))
    print("Learning rate gan = %s" % (tl.global_flag['paramLR_GAN']))
    print("Learning rate decay = %s" % (tl.global_flag['paramLD']))
    print("Decay init learning rate after epochs = %s" % (tl.global_flag['decayEpochs']))
    print("Decay GAN learning rate after epochs = %s" % (tl.global_flag['decayEpochsGAN']))
    print("GAN training = %s" % (tl.global_flag['GAN']))
    print("Number of epochs for GAN training = %s" % (tl.global_flag['nepochs_GAN']))
    print("Read in batches = %s" % (tl.global_flag['readBatch_flag']))
    print("Full path/name of the generative network to be evaluated = %s" % (tl.global_flag['testModel']))

    if tl.global_flag['mode'] == 'train':
        train()
        # evaluate(afterTrain=True)
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate(afterTrain=False)
    else:
        raise Exception("Unknow --mode")
