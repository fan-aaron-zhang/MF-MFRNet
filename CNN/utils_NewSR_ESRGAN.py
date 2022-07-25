#! /usr/bin/python
# -*- coding: utf8 -*-

###################################################################################################
#### This code was developed by Mariana Afonso, Phd student @ University of Bristol, UK, 2018 #####
################################## All rights reserved © ##########################################

from tensorlayer.prepro import *
import scipy
import numpy as np
import struct
import matplotlib.pyplot as plt
import scipy.misc
import random
from skimage.restoration import denoise_nl_means, estimate_sigma
import sys
import imageio
import math
from skimage.feature import greycomatrix, greycoprops


def create_training_images(listHR, pathHR, listLR, pathLR, ratio=2, block_size=96, batch_size=16, rotate=1, save_folder='', n_threads=32, readBatch_flag=1, createValid=0):
    """ Returns all images in array by given path and name of each image file. """

    # check if train dataset folder already exists and if so, do not create it again
    if os.path.isdir(os.path.join(save_folder)) == False:

        tl.files.exists_or_mkdir(os.path.join(save_folder))
        tl.files.exists_or_mkdir(os.path.join(save_folder, 'HR'))
        tl.files.exists_or_mkdir(os.path.join(save_folder, 'LR'))
        tl.files.exists_or_mkdir(os.path.join(save_folder, 'png'))
        tl.files.exists_or_mkdir(os.path.join(save_folder, "batch_blocks"))
        tl.files.exists_or_mkdir(os.path.join(save_folder, "batch_blocks", "HR"))
        tl.files.exists_or_mkdir(os.path.join(save_folder, "batch_blocks", "LR"))

        #np.random.seed(seed=10) # setting a seed for the validation set so that it creates the same blocks for
                                # VDSR and SRResNet

        for idx in range(0, len(listHR), n_threads):
            hr_imgs_list = listHR[idx: idx + n_threads]
            lr_imgs_list = listLR[idx: idx + n_threads]
            tl.prepro.threading_data2(hr_imgs_list, lr_imgs_list, fn=save_blocks_fn, pathHR=pathHR, pathLR=pathLR,
                                      ratio=ratio, block_size=block_size, rotate=rotate, save_folder=save_folder, createValid=createValid)
            # b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_frames_fn, path=path)
            # imgs.extend(b_imgs)
            print('read %d input videos' % (idx + n_threads))

        if readBatch_flag == 1:
            # create files with an aggregate with batch_size size

            blocksPathHR1 = os.path.join(save_folder,"HR")
            blocksPathLR1 = os.path.join(save_folder,"LR")
            batchBlocksPathHR = os.path.join(save_folder, "batch_blocks", "HR")
            batchBlocksPathLR = os.path.join(save_folder, "batch_blocks", "LR")

            blockFilesHR = [os.path.join(blocksPathHR1, f) for f in os.listdir(blocksPathHR1) if os.path.isfile(os.path.join(blocksPathHR1, f))]
            blockFilesLR = [os.path.join(blocksPathLR1, f) for f in os.listdir(blocksPathLR1) if os.path.isfile(os.path.join(blocksPathLR1, f))]

            print("total number of HR blocks = " + str(len(blockFilesHR)))
            print("total number of LR blocks = " + str(len(blockFilesLR)))

            assert len(blockFilesHR) == len(blockFilesLR), "Number of blocks between HR and LR are not consistent"

            #blockFilesHR, blockFilesLR = unison_shuffle(blockFilesHR, blockFilesLR)

            # parameters
            widthHR = int(block_size * ratio)
            heightHR = int(block_size * ratio)
            widthLR = block_size
            heightLR = block_size

            sizeFrameHR = widthHR * heightHR
            sizeFrameLR = widthLR * heightLR

            sizeBytes = os.path.getsize(os.path.join(save_folder, "HR", blockFilesHR[0]))

            multiplier = int(sizeBytes / (sizeFrameHR * 3))

            if multiplier == 1:
                elementType = 'B'
            elif multiplier == 2:
                elementType = 'H'
            else:
                raise Exception("Error reading file: multiplier should be either 1 or 2")

            buf_batchHR = []
            buf_batchLR = []
            batch_count = 0
            for i in range(0, len(blockFilesHR)):

                with open(blockFilesHR[i], 'rb') as fileID:

                    buf = struct.unpack(str(3 * sizeFrameHR) + elementType, fileID.read(3 * sizeFrameHR * multiplier))

                    buf_batchHR.extend(buf)

                    if i != 0 and (i + 1) % batch_size == 0:
                        blockFilename = os.path.join(batchBlocksPathHR, str(batch_count) + ".image")
                        with open(blockFilename, 'ab') as fileSaveID:
                            write_pixels(np.asarray(buf_batchHR), len(buf_batchHR), multiplier, fileSaveID)

                        buf_batchHR = []

                with open(blockFilesLR[i], 'rb') as fileID:

                    buf = struct.unpack(str(3 * sizeFrameLR) + elementType, fileID.read(3 * sizeFrameLR * multiplier))

                    buf_batchLR.extend(buf)

                    if i != 0 and (i + 1) % batch_size == 0:
                        blockFilename = os.path.join(batchBlocksPathLR,str(batch_count) + ".image")
                        with open(blockFilename, 'ab') as fileSaveID:
                            write_pixels(np.asarray(buf_batchLR), len(buf_batchLR), multiplier, fileSaveID)

                        buf_batchLR = []
                        batch_count = batch_count + 1

        return 0

    else:
        return -1


def read_blocks(img_list, path='', mode='HR', ratio=2, block_size=96, n_threads=16, dimensions=1):
    """ Returns all images in array by given path and name of each image file. """
    imgs_all = []
    for idx in range(0, len(img_list), n_threads):
        img_list_curr = img_list[idx: idx + n_threads]
        imgs = tl.prepro.threading_data(img_list_curr, fn=get_blocks_fn, path=path, HRorLR=mode, ratio=ratio,
                                        block_size=block_size, dimensions=dimensions)
        imgs_all.extend(imgs)

    return imgs_all

def read_blocks_batch(img_name, path='', mode='HR', ratio=2, block_size=96, dimensions=1, batch_size=64):
    """ Returns all images in array by given path and name of each image file. """

    imgs_all = get_blocks_batch_fn(img_name, path=path, HRorLR=mode, ratio=ratio,
                                        block_size=block_size, dimensions=dimensions, batch_size=batch_size)

    return imgs_all


def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn_384(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def crop_sub_imgs_fn(x, w_new, h_new, is_random=True):
    x = crop(x, wrg=w_new, hrg=h_new, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def loadYUVfile(filename, width, height, idxFrames, colorSampling, bitDepth):

    numFrames = idxFrames.size

    if bitDepth == 8:
        multiplier = 1
        elementType = 'B'
    elif bitDepth == 10:
        multiplier = 2
        elementType = 'H'
    else:
        raise Exception("Error reading file: bit depth not allowed (8 or 10 )")

    if colorSampling == 420:
        sizeFrame = 1.5 * width * height * multiplier
        width_size = int(width / 2)
        height_size = int(height / 2)
    elif colorSampling == 422:
        sizeFrame = 2 * width * height * multiplier
        width_size = int(width / 2)
        height_size = height
    elif colorSampling == 444:
        sizeFrame = 3 * width * height * multiplier
        width_size = width
        height_size = height
    else:
        raise Exception("Error reading file: color sampling not allowed (420, 422 or 444 )")

    sizeY = width * height
    sizeColor = width_size * height_size 

    with open(filename,'rb') as fileID:

        fileID.seek(int(idxFrames[0]*sizeFrame),0)

        for iframe in range (0,numFrames):

            try:
                buf = struct.unpack(str(sizeY)+elementType, fileID.read(sizeY*multiplier))
            except:
                return np.array(1)
            buf = np.asarray(buf)
            bufY = np.reshape(buf, (height,width))

            try:
                buf = struct.unpack(str(sizeColor)+elementType, fileID.read(sizeColor*multiplier))
            except:
                return np.array(1)
            buf = np.asarray(buf)
            bufU = np.reshape(buf, (height_size,width_size))

            try:
                buf = struct.unpack(str(sizeColor)+elementType, fileID.read(sizeColor*multiplier))
            except:
                return np.array(1)
            buf = np.asarray(buf)
            bufV = np.reshape(buf, (height_size,width_size))

            if colorSampling == 420:
                bufU = bufU.repeat(2, axis=0).repeat(2, axis=1)
                bufV = bufV.repeat(2, axis=0).repeat(2, axis=1)
            elif colorSampling == 422:
                bufU = bufU.repeat(2, axis=1)
                bufV = bufV.repeat(2, axis=1)

            image = np.stack((bufY,bufU,bufV), axis=2)
            image.resize((1,height,width,3))

            if iframe == 0:
                video = image
            else:
                video = np.concatenate((video,image), axis=0)

    return video

def saveYUVfile(video, filename, mode, colorSampling, bitDepth):

    if bitDepth == 8:
        multiplier = 1
    elif bitDepth == 10:
        multiplier = 2
    else: 
        raise Exception("Error writing file: bit depth not allowed (8 or 10)")

    if mode != 'ba' and mode != 'bw' and mode != 'ab' and mode != 'wb':
        raise Exception("Error writing file: writing mode not allowed ('ab' or 'wb')")

    #fileID = open(filename, mode)

    numFrames = video.shape[0]
    height = video.shape[1]
    width = video.shape[2]
    frameSize = height*width

    if colorSampling != 'Y':

        if colorSampling == 420:
            sampling_width = 2
            sampling_height = 2
        elif colorSampling == 422:
            sampling_width = 2
            sampling_height = 1
        elif colorSampling == 444:
            sampling_width = 1
            sampling_height = 1
        else:
            raise Exception("Error reading file: color sampling not allowed (420, 422 or 444 )")

        frameSizeColor = int((height/sampling_height)*(width/sampling_width))

    with open(filename, mode) as fileID:

        for iframe in range(0,numFrames):

            imageY = video[iframe,:,:,0].reshape((height*width))

            write_pixels(imageY, frameSize, multiplier, fileID)

            if colorSampling != 'Y':

                imageU = video[iframe,0:height:sampling_height,0:width:sampling_width,1].reshape((frameSizeColor))

                write_pixels(imageU, frameSizeColor, multiplier, fileID)

                imageV = video[iframe,0:height:sampling_height,0:width:sampling_width,2].reshape((frameSizeColor))

                write_pixels(imageV, frameSizeColor, multiplier, fileID)

    return 0

def write_pixels(image, frameSize, multiplier, fileID):
    
    if multiplier == 1:
        elementType = 'B'
    elif multiplier == 2:
        elementType = 'H'
    else:
        raise Exception("Error reading file: multiplier not allowed (1 or 2)")
	
    for i in range(0,frameSize):
        pixel_bytes = image[i]
        try:
            pixel_bytes = struct.pack(elementType, pixel_bytes)
            fileID.write(pixel_bytes)
        except:
            print(pixel_bytes)
            print(elementType)
            sys.exit("Error: struck.pack")
        # pixel_bytes = struct.pack(str(sizeFrame)+elementType, fileID.read(sizeFrame*multiplier))
        # pixel_bytes = int(pixel_bytes).to_bytes(multiplier, byteorder='little', signed=False)
        # fileID.write(pixel_bytes)

# crop several image patches on a grid from an input image 
def crop_grid(x, wrg, hrg, row_index=0, col_index=1, channel_index=2):
    """Randomly or centrally crop an image.

    Parameters
    ----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    wrg : float
        Size of weight.
    hrg : float
        Size of height.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    """
    h, w = x.shape[row_index], x.shape[col_index]
    assert (h > hrg) and (w > wrg), "The size of cropping should smaller than the original image"

    h_num = int(np.floor(float(h)/hrg))
    w_num = int(np.floor(float(w)/wrg)) 

    results = []

    for i in range(0,h_num):
        for j in range(0,w_num):

            h_offset = i*hrg
            w_offset = j*wrg
            h_end = h_offset + hrg
            w_end = w_offset + wrg

            results.append( x[h_offset: h_end, w_offset: w_end])
    
    return results

def crop_grid_random(frameHR, frameLR, ratio, block_size, bitDepth, createValid=0, row_index=0, col_index=1, channel_index=2):
    """Randomly or centrally crop an image.
    """

    if createValid == 1:
        nblocks = 16 # 16 blocks for validation dataset
    else:
        nblocks = 32

    maxValue = float(2 ** bitDepth)

    h_HR, w_HR = frameHR.shape[row_index], frameHR.shape[col_index]
    h_LR, w_LR = frameLR.shape[row_index], frameLR.shape[col_index]
    assert (h_HR == ratio*h_LR) and (w_HR == ratio*w_LR), "The size of the HR frame should be exactly ratio times of the LR frame"

    hrg_LR = block_size
    wrg_LR = block_size
    hrg_HR = int(block_size*ratio)
    wrg_HR = int(block_size*ratio)

    assert (h_HR > hrg_HR) and (w_HR > wrg_HR) and (h_LR > hrg_LR) and (w_LR > wrg_LR), "The size of cropping should smaller than the original image"

    blocksHR = []
    blocksLR = []
    skip_flag = 0

    frac_ratio = ratio-float(int(ratio))
    if frac_ratio == 0:
        frac_ratio = 1

    ### use this to generate all blocks from a frame
    # for i in range(0,int(h_LR/block_size)):
    #     for j in range(0,int(w_LR/block_size)):
    #
    #         h_offset_LR = i*block_size
    #         w_offset_LR = j*block_size
    #         if frac_ratio != 0:
    #             h_offset_LR = int(int(h_offset_LR*frac_ratio)/frac_ratio)
    #             w_offset_LR = int(int(w_offset_LR*frac_ratio)/frac_ratio)
    #
    #         h_offset_HR = int(h_offset_LR*ratio)
    #         w_offset_HR = int(w_offset_LR*ratio)
    #
    #         h_end_HR = h_offset_HR + hrg_HR
    #         w_end_HR = w_offset_HR + wrg_HR
    #
    #         h_end_LR = h_offset_LR + hrg_LR
    #         w_end_LR = w_offset_LR + wrg_LR
    #
    #         assert (h_end_HR <= h_HR) and (w_end_HR <= w_HR) and (h_end_LR <= h_LR) and (w_end_LR <= w_LR), "The block cropping indexes should be smaller than the original image"
    #
    #         blocksHR.append(frameHR[h_offset_HR: h_end_HR, w_offset_HR: w_end_HR])
    #         blocksLR.append(frameLR[h_offset_LR: h_end_LR, w_offset_LR: w_end_LR])

    for i in range(0,nblocks):

        ### enable when creating the training dataset
        if createValid == 0:
            mean_squares_sobel = 0
            count = 0
            while mean_squares_sobel < 0.01 and count < 50:

                h_offset_LR = int(np.random.uniform(0, h_LR-hrg_LR))
                w_offset_LR = int(np.random.uniform(0, w_LR-wrg_LR))
                h_offset_LR = int(int(h_offset_LR*frac_ratio)/frac_ratio)
                w_offset_LR = int(int(w_offset_LR*frac_ratio)/frac_ratio)
                Y_LR = frameLR[h_offset_LR: h_offset_LR + hrg_LR, w_offset_LR: w_offset_LR + wrg_LR, 0]
                Y_sobel = scipy.ndimage.sobel(normalize_array(Y_LR, bitDepth))
                mean_squares_sobel = np.mean(np.square(Y_sobel))

                # Y_sobel_x = scipy.ndimage.sobel(normalize_array(Y_LR, bitDepth), axis=0)
                # Y_sobel_y = scipy.ndimage.sobel(normalize_array(Y_LR, bitDepth), axis=1)
                # mag_sobel = np.sqrt(np.power(Y_sobel_x,2) + np.power(Y_sobel_y,2))
                # dir_sobel = np.arctan(Y_sobel_y/Y_sobel_x)
                # hist_mag, bin_edges_sobel = np.divide(np.histogram(mag_sobel,bins=64),(hrg_LR*wrg_LR))
                # log2sobel = np.log2(hist_mag)
                # log2sobel[np.isinf(log2sobel)] = 0
                # entropy_mag = -np.sum(hist_mag*log2sobel)
                # hist_dir, bin_edges_sobel = np.divide(np.histogram(dir_sobel[~np.isnan(dir_sobel)], bins=64),(hrg_LR*wrg_LR))
                # log2sobel = np.log2(hist_dir)
                # log2sobel[np.isinf(log2sobel)] = 0
                # entropy_dir = -np.sum(hist_dir * log2sobel)

                count = count + 1
                if count == 50:
                    print("skipping block...")
                    skip_flag = 1

            if skip_flag==1:
                skip_flag = 0
                continue

            ### enable when creating the training dataset
            h_offset_HR = int(h_offset_LR * ratio)
            w_offset_HR = int(w_offset_LR * ratio)

            h_end_HR = h_offset_HR + hrg_HR
            w_end_HR = w_offset_HR + wrg_HR

            h_end_LR = h_offset_LR + hrg_LR
            w_end_LR = w_offset_LR + wrg_LR

            assert (h_end_HR < h_HR) and (w_end_HR < w_HR) and (h_end_LR < h_LR) and (
                    w_end_LR < w_LR), "The block cropping indexes should be smaller than the original image"

            blocksHR.append(frameHR[h_offset_HR: h_end_HR, w_offset_HR: w_end_HR])
            blocksLR.append(frameLR[h_offset_LR: h_end_LR, w_offset_LR: w_end_LR])

        else:
            ## enable this when creating validation dataset
            h_offset_HR = int(np.random.uniform(0, h_HR - hrg_HR))
            w_offset_HR = int(np.random.uniform(0, w_HR - wrg_HR))
            h_offset_HR = int(int(h_offset_HR / (ratio/frac_ratio)) * (ratio/frac_ratio))
            w_offset_HR = int(int(w_offset_HR / (ratio/frac_ratio)) * (ratio/frac_ratio))

            block_HR = frameHR[h_offset_HR: h_offset_HR + hrg_HR, w_offset_HR: w_offset_HR + wrg_HR]

            h_offset_LR = int(h_offset_HR/ratio)
            w_offset_LR = int(w_offset_HR/ratio)

            block_LR = frameLR[h_offset_LR: h_offset_LR + hrg_LR, w_offset_LR: w_offset_LR + wrg_LR]

            h_end_HR = h_offset_HR + hrg_HR
            w_end_HR = w_offset_HR + wrg_HR

            h_end_LR = h_offset_LR + hrg_LR
            w_end_LR = w_offset_LR + wrg_LR

            assert (h_end_HR < h_HR) and (w_end_HR < w_HR) and (h_end_LR < h_LR) and (w_end_LR < w_LR), "The block cropping indexes should be smaller than the original image"

            blocksHR.append(block_HR)
            blocksLR.append(block_LR)

        #### enable this code to split between textures and structures
        #  Y_HR = frameHR[h_offset_HR: h_end_HR, w_offset_HR: w_end_HR, 0]
        #
        # glcm = greycomatrix(Y_HR, [16], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=int(maxValue+1), symmetric=True, normed=True)
        # glcm_cont = greycoprops(glcm, 'contrast')[0, 0]
        # glcm_diss = greycoprops(glcm, 'dissimilarity')[0, 0]
        # glcm_hom = greycoprops(glcm, 'homogeneity')[0, 0]
        # glcm_corr = greycoprops(glcm, 'correlation')[0, 0]
        #
        # Y_HR_sobel = scipy.ndimage.sobel(normalize_array(Y_HR, bitDepth))
        #
        # thres_sobel = 0.01
        # sobel_larger = Y_HR_sobel>thres_sobel
        # sum_sobel = np.sum(sobel_larger)
        # edgeness = sum_sobel/(block_size*ratio*block_size*ratio)
        #
        # if edgeness >= 0.45 and glcm_cont <= 15000:
        #     #if glcm_hom <= 0.03 and glcm_corr <= 0.25:
        #     print("texture")
        #     #imageio.imwrite("texture/block_HR_c%.1f" % glcm_cont + "_d%.1f" % glcm_diss + "_h%.3f" % glcm_hom + "_cr%.1f" % glcm_corr + "_edges%.2f.jpg" % edgeness, normalize_array(Y_HR, bitDepth))
        # else:
        #     print("structure")
        #     blocksHR.append(frameHR[h_offset_HR: h_end_HR, w_offset_HR: w_end_HR])
        #     blocksLR.append(frameLR[h_offset_LR: h_end_LR, w_offset_LR: w_end_LR])
        #     #imageio.imwrite("structure/block_HR_c%.1f" % glcm_cont + "_d%.1f" % glcm_diss + "_h%.3f" % glcm_hom + "_cr%.1f" % glcm_corr + "_edges%.2f.jpg" % edgeness, normalize_array(Y_HR, bitDepth))
    
    return blocksHR, blocksLR

# def crop_multiple(image_list, wcrop, hcrop):
#
#     cropped_ims = []
#
#     for i in range(0, len(image_list)):
#         image = image_list[i]
#         cropped_ims.extend(crop_grid_16blocks(image, wcrop, hcrop))
#
#     return cropped_ims

class videoParams:
    filename = ""
    seqName = ""
    width = 0
    height = 0
    colorSampling = 0
    bitDepth = 0
    frameRate = 0

    def extractVideoParameters(self):
        splitstr = self.filename.split('_')
        self.seqName = splitstr[0]
        resolution = splitstr[1]
        resolution = resolution.split('x')
        self.width = int(resolution[0])
        self.height = int(resolution[1])
        self.frameRate = splitstr[2]
        self.frameRate = int(self.frameRate[0:-3])
        self.bitDepth = splitstr[3]
        self.bitDepth = int(self.bitDepth[0:-3])
        self.outBitDepth = 10
        colorsamplingstr = splitstr[4]
        colorsamplingstr = colorsamplingstr.split('.')
        self.colorSampling = int(colorsamplingstr[0])

        return 0

    def printParams(self):
        print('Video parameters::::')
        print('file name: ' + self.filename)
        print('sequence name: ' + self.seqName)
        print('width: ' + str(self.width) + ' height: ' + str(self.height))
        print('bitdepth: ' + str(self.bitDepth))
        print('color sampling: ' + str(self.colorSampling))
        print('frame rate: ' + str(self.frameRate))
        return 0

def read_all_frames(img_list, path):

    nframes = 16

    all_frames = []    

    for i in range(0,len(img_list)):

        filename = img_list[i]

        myParams = videoParams()

        myParams.filename = filename 
        myParams.extractVideoParameters()
        myParams.printParams()

        if myParams.frameRate == 24 or myParams.frameRate == 25 or myParams.frameRate == 30:
            step1 = 1
            step2 = 3
        elif myParams.frameRate == 50:
            step1 = 3
            step2 = 3
        elif myParams.frameRate == 60:
            step1 = 3
            step2 = 5
        elif myParams.frameRate == 120:
            step1 = 7
            step2 = 9
        else:
            raise Exception("frame rate not either 24, 25, 30, 60 or 120")

        frame_pos = 0

        for iframe in range(0,nframes):

            frame = loadYUVfile(os.path.join(path,myParams.filename), myParams.width, myParams.height, np.array([frame_pos]), myParams.colorSampling, myParams.bitDepth)
            
            rgb_image = yuv2rgb(np.squeeze(frame),myParams.bitDepth)
            #imageio.imwrite('outfile.jpg', rgb_image)

            print('reading frame ' + str(frame_pos))

            if iframe % 2 == 0: 
                frame_pos = frame_pos + step1
            else:
                frame_pos = frame_pos + step2

            print("frame dimensions = " + str(rgb_image.shape))

            all_frames.append(rgb_image)

            print("Total number of frames read = " + str(len(all_frames)))

    return all_frames

def save_blocks_fn(filenameHR, filenameLR, pathHR, pathLR, ratio, block_size, rotate, save_folder, createValid):

    nframes = 8

    if rotate == 1:
        max_rot = 4
    else:
        max_rot = 1

    #all_frames = []    

    myParamsHR = videoParams()
    myParamsHR.filename = filenameHR 
    myParamsHR.extractVideoParameters()
    #myParamsHR.printParams()

    myParamsLR = videoParams()
    myParamsLR.filename = filenameLR 
    myParamsLR.extractVideoParameters()

    if myParamsHR.frameRate == 24 or myParamsHR.frameRate == 25 or myParamsHR.frameRate == 30:
        step1 = 3
        step2 = 5
    elif myParamsHR.frameRate == 50:
        step1 = 7
        step2 = 9
        # step1 = 5
        # step2 = 7 # todo: uncomment
    elif myParamsHR.frameRate == 60:
        step1 = 7
        step2 = 9
    elif myParamsHR.frameRate == 120:
        # step1 = 15
        # step2 = 17 # todo: uncomment
        step1 = 7
        step2 = 9
    else:
        raise Exception("frame rate not either 24, 25, 30, 60 or 120")

    # enable this for creating the validation dataset
    if createValid == 1:
        step1 = 37
        step2 = 47

    frame_pos = 0

    for iframe in range(0,nframes):

        frameHR = np.squeeze(loadYUVfile(os.path.join(pathHR,filenameHR), myParamsHR.width, myParamsHR.height, np.array([frame_pos]), myParamsHR.colorSampling, myParamsHR.bitDepth))
        frameLR = np.squeeze(loadYUVfile(os.path.join(pathLR,filenameLR), myParamsLR.width, myParamsLR.height, np.array([frame_pos]), myParamsLR.colorSampling, myParamsLR.bitDepth))

        if iframe % 2 == 0: 
            frame_pos = frame_pos + step1
        else:
            frame_pos = frame_pos + step2
            
        blocksHR, blocksLR = crop_grid_random(frameHR, frameLR, ratio, block_size, myParamsHR.bitDepth, createValid)

        ### enable when rotating each block only once
        rot = 0
        for block in range(0,len(blocksHR)):
            # for rot in range(0,max_rot): ### disble when when rotating each block only once

            # currentBlockHR = blocksHR[block]
            # currentBlockHR = yuv2rgb(np.squeeze(currentBlockHR),myParamsHR.bitDepth)
            # currentBlockHR = normalize_array(currentBlockHR, myParamsHR.bitDepth)
            # currentBlockHR = denoise_image(currentBlockHR)
            # currentBlockHR = inverse_normalize_array(currentBlockHR, myParamsHR.bitDepth)
            # currentBlockHR = rgb2yuv(np.squeeze(currentBlockHR),myParamsHR.bitDepth)
            # currentBlockHR = np.int32(currentBlockHR)

            currBlockHR = blocksHR[block]
            currBlockLR = blocksLR[block]

            if rot != 0:
                currBlockHR = np.rot90(currBlockHR,rot)
                currBlockLR = np.rot90(currBlockLR,rot)

            save_block(currBlockHR,myParamsHR.seqName,iframe,block,rot,myParamsHR.bitDepth,save_folder,'HR')
            save_block(currBlockLR,myParamsHR.seqName,iframe,block,rot,myParamsHR.bitDepth,save_folder,'LR')

            rgb_block = yuv2rgb(np.squeeze(currBlockHR),myParamsHR.bitDepth)
            rgb_block = normalize_array(rgb_block, myParamsHR.bitDepth)
            outfilename = myParamsHR.seqName + '_frame' + str(iframe+1) + '_block' + str(block) + '_rot' + str(rot) + '_' + 'HR' + '.png'
            outfilename = os.path.join(save_folder,'png', outfilename)
            imageio.imwrite(outfilename, np.uint8(255*rgb_block))

            rgb_block = yuv2rgb(np.squeeze(currBlockLR),myParamsHR.bitDepth)
            rgb_block = normalize_array(rgb_block, myParamsHR.bitDepth)
            outfilename = myParamsHR.seqName + '_frame' + str(iframe+1) + '_block' + str(block) + '_rot' + str(rot) + '_' + 'LR' + '.png'
            outfilename = os.path.join(save_folder,'png',outfilename)
            imageio.imwrite(outfilename, np.uint8(255*rgb_block))

            # upsample and save as image file to compare

            # rgb_block_up = imresize(np.uint8(255 * rgb_block), size=[int(block_size * ratio), int(block_size * ratio)],
            #                         interp='bicubic', mode=None)
            # outfilename = myParamsHR.seqName + '_frame' + str(iframe + 1) + '_block' + str(block) + '_rot' + str(
            #     rot) + '_' + 'UP' + '.png'
            # outfilename = os.path.join(save_folder, 'png', outfilename)
            # imageio.imwrite(outfilename, rgb_block_up)

            ### enable when rotating each block only once
            rot = rot + 1
            if rot == 4 or createValid == 1:
                rot = 0


def save_block(block_im,seqName,iframe,block,rot,bitDepth,save_folder,HRorLR):
    
    if bitDepth == 8:
        multiplier = 1
    elif bitDepth == 10:
        multiplier = 2
    else: 
        raise Exception("Error writing file: bit depth not allowed (8 or 10)")
    
    blockY = block_im[:,:,0]
    blockU = block_im[:,:,1]
    blockV = block_im[:,:,2]

    blockY = blockY.flatten()
    blockU = blockU.flatten()
    blockV = blockV.flatten()

    block_im = np.append(blockY,blockU)
    block_im = np.append(block_im,blockV)
    
    filename = seqName + '_frame' + str(iframe+1) + '_block' + str(block) + '_rot' + str(rot) + '_'+ HRorLR + '.image'
    
    with open(os.path.join(save_folder,HRorLR,filename), 'wb') as fileID:
        write_pixels(block_im, len(block_im), multiplier, fileID)


def save_mean_block(block_im, bitDepth, save_folder, HRorLR):
    if bitDepth == 8:
        multiplier = 1
    elif bitDepth == 10:
        multiplier = 2
    else:
        raise Exception("Error writing file: bit depth not allowed (8 or 10)")

    blockY = block_im[:, :, 0]
    blockU = block_im[:, :, 1]
    blockV = block_im[:, :, 2]

    blockY = blockY.flatten()
    blockU = blockU.flatten()
    blockV = blockV.flatten()

    block_im = np.append(blockY, blockU)
    block_im = np.append(block_im, blockV)

    filename = 'mean_block_' + HRorLR + '.image'

    with open(os.path.join(save_folder, filename), 'wb') as fileID:
        write_pixels(block_im, len(block_im), multiplier, fileID)

def get_blocks_fn(filename, path, HRorLR, ratio, block_size, dimensions):

    if HRorLR == 'HR':
        width = int(block_size*ratio)
        height = int(block_size*ratio)
    elif HRorLR == 'LR':
        width = block_size
        height = block_size
    else:
        raise Exception("HRorLR should be either HR or LR")

    sizeFrame = width*height

    sizeBytes = os.path.getsize(os.path.join(path,filename))

    multiplier =  int(sizeBytes / (sizeFrame*3))

    if multiplier == 1:
        bitDepth = 8
        elementType = 'B'
    elif multiplier == 2:
        bitDepth = 10
        elementType = 'H'
    else:
        raise Exception("Error reading file: multiplier should be either 1 or 2")

    with open(os.path.join(path,filename),'rb') as fileID:

        buf = struct.unpack(str(sizeFrame)+elementType, fileID.read(sizeFrame*multiplier))
        bufY = np.reshape(np.asarray(buf), (height,width))

        buf = struct.unpack(str(sizeFrame)+elementType, fileID.read(sizeFrame*multiplier))
        bufU = np.reshape(np.asarray(buf), (height,width))

        buf = struct.unpack(str(sizeFrame)+elementType, fileID.read(sizeFrame*multiplier))
        bufV = np.reshape(np.asarray(buf), (height,width))

        block = np.stack((bufY,bufU,bufV), axis=2)

        if (dimensions == 1):
            block = block[:, :, 0]  # keep only the Y dimension
        else:
            block = yuv2rgb(block, bitDepth)  # I think this is wrong because I am dividing by 4 here and then dividing again by the bitdepthmultiplier when I normalize

        block = normalize_array(block, bitDepth)

        # if HRorLR == 'HR':
        #     block = denoise_image(block)

    return block

class listMem:
    n_read = 0
    pos = 0
    idx = 0
reading = dict()
def get_blocks_batch_fn(filelist, path, HRorLR, ratio, block_size, dimensions, batch_size): # by Zihao. filename -> filelist, and maintain a global index 'f_idx'
    global reading
    
    if id(filelist) not in reading:
        reading[id(filelist)] = listMem()
    
    if len(filelist)==0: # by Zihao. Call this by the end of each epoch to reset these global variables
        reading = dict()
        return
    
    if HRorLR == 'HR':
        width = int(block_size*ratio)
        height = int(block_size*ratio)
    elif HRorLR == 'LR':
        width = block_size
        height = block_size
    else:
        raise Exception("HRorLR should be either HR or LR")

    sizeFrame = width*height

    sizeBytes = os.path.getsize(os.path.join(path,filelist[reading[id(filelist)].idx]))

    #multiplier =  int(sizeBytes / (batch_size*sizeFrame*3))
    multiplier = 2 # by Zihao for batch_size=8

    if multiplier == 1:
        bitDepth = 8
        elementType = 'B'
    elif multiplier == 2:
        bitDepth = 10
        elementType = 'H'
    else:
        raise Exception("Error reading file: multiplier should be either 1 or 2")

    all_blocks = []
    fileID = open(os.path.join(path,filelist[reading[id(filelist)].idx]),'rb')
    fileID.seek(reading[id(filelist)].pos)
    for i in range(0,batch_size):
        buf = struct.unpack(str(sizeFrame)+elementType, fileID.read(sizeFrame*multiplier))
        bufY = np.reshape(np.asarray(buf), (height,width))

        buf = struct.unpack(str(sizeFrame)+elementType, fileID.read(sizeFrame*multiplier))
        bufU = np.reshape(np.asarray(buf), (height,width))

        buf = struct.unpack(str(sizeFrame)+elementType, fileID.read(sizeFrame*multiplier))
        bufV = np.reshape(np.asarray(buf), (height,width))

        block = np.stack((bufY,bufU,bufV), axis=2)

        if dimensions == 1:
            block = block[:, :, 0]  # keep only the Y dimension
        #else:
        #    block = yuv2rgb(block, bitDepth) # convert to RGB as the input

        block = normalize_array(block, bitDepth)

        all_blocks.append(block)
        
        # check if reach the file end
        reading[id(filelist)].n_read += 1
        if reading[id(filelist)].n_read >= 16:
            reading[id(filelist)].n_read = 0
            reading[id(filelist)].idx += 1
            fileID.close()
            fileID = open(os.path.join(path,filelist[reading[id(filelist)].idx]),'rb')
        if i == batch_size-1: # finish reading this batch
            reading[id(filelist)].pos = fileID.tell()
            fileID.close()
    
    #logfile = open('./test.out','a+') # by Zihao
    #logfile.write('Size of Files:\n')
    #logfile.write('LRCnt: {} KB\n'.format(sys.getsizeof(LRCnt)//1024))
    #logfile.write('HRCnt: {} KB\n'.format(sys.getsizeof(HRCnt)//1024))
    #logfile.close()

    return all_blocks

def denoise_image(image):
    # denoise image using non-local means denoising implemented in package skimage

    sigma_est = np.mean(estimate_sigma(image,multichannel=True))
    denoise_im = denoise_nl_means(image, h=2*sigma_est, fast_mode=True, patch_size=5, patch_distance=6, multichannel=True)

    # find out when any pixel is nan or inf (which is wrong) so that it can be replaced by a finite number
    idxNan = np.argwhere(np.isnan(denoise_im))
    idxInf = np.argwhere(np.isinf(denoise_im))
    if idxNan.any() or idxInf.any():
        denoise_im = np.nan_to_num(denoise_im)
        
    # make sure that the denoising process does not overflow the input range of the image [0,1]
    denoise_im[denoise_im < 0] = 0
    denoise_im[denoise_im > 1] = 1

    return denoise_im

def yuv2rgb_multiple(array, bitdepth):

    rgbarray = []
    for idxIm in range(len(array)):

        image = np.squeeze(array[idxIm])
        image_rgb = yuv2rgb(image, bitdepth)

        rgbarray.append(image_rgb)

    return np.asarray(rgbarray)

def yuv2rgb(image, bitDepth):

    N = ((2**bitDepth)-1)

    Y = np.float32(image[:,:,0])
    
    U = np.float32(image[:,:,1])
    
    V = np.float32(image[:,:,2])

    Y = Y/N
    U = U/N
    V = V/N

    fy = Y
    fu = U-0.5
    fv = V-0.5

    # parameters
    KR = 0.2627
    KG = 0.6780
    KB = 0.0593 

    R = fy + 1.4746*fv
    B = fy + 1.8814*fu
    G = -(B*KB+KR*R-Y)/KG

    R[R<0] = 0
    R[R>1] = 1
    G[G<0] = 0
    G[G>1] = 1
    B[B<0] = 0
    B[B>1] = 1

    rgb_image = np.array([R,G,B])
    rgb_image = np.swapaxes(rgb_image,0,2)
    rgb_image = np.swapaxes(rgb_image,0,1)
    rgb_image = rgb_image*N

    return rgb_image

def rgb2yuv(image, bitDepth):

    N = ((2**bitDepth)-1)

    R = np.float32(image[:,:,0])
    G = np.float32(image[:,:,1])
    B = np.float32(image[:,:,2])

    R = R/N
    G = G/N
    B = B/N

    # parameters
    KR = 0.2627
    KG = 0.6780
    KB = 0.0593 

    Y = KR*R + KG*G + KB*B
    U = (B-Y)/1.8814
    V = (R-Y)/1.4746

    U = U+0.5
    V = V+0.5

    Y[Y<0] = 0
    Y[Y>1] = 1
    U[U<0] = 0
    U[U>1] = 1
    V[V<0] = 0
    V[V>1] = 1

    yuv_image = np.array([Y,U,V])
    yuv_image = np.swapaxes(yuv_image,0,2)
    yuv_image = np.swapaxes(yuv_image,0,1)
    yuv_image = yuv_image*N

    return yuv_image

def yuv2rgb2(image, bitDepth):

    if bitDepth == 10:
        div = 4
    else:
        div = 1

    Y = np.float32(image[:,:,0]/div)
    
    U = np.float32(image[:,:,1]/div)
    
    V = np.float32(image[:,:,2]/div)
    
    # B = 1.164*(Y - 16) + 2.018*(U - 128)
    # G = 1.164*(Y - 16) - 0.813*(V - 128) - 0.391*(U - 128)
    # R = 1.164*(Y - 16) + 1.596*(V - 128)

    R = Y + 1.40200 * (V - 128)
    G = Y - 0.34414 * (U - 128) - 0.71414 * (V - 128)
    B = Y + 1.77200 * (U - 128)

    rgb_image = np.array([R,G,B])
    rgb_image = np.swapaxes(rgb_image,0,2)
    rgb_image = np.swapaxes(rgb_image,0,1)

    # print("Y max = " + str(np.amax(Y)) + ", Y min = " + str(np.amin(Y)))
    # print("U max = " + str(np.amax(U)) + ", U min = " + str(np.amin(U)))
    # print("V max = " + str(np.amax(V)) + ", V min = " + str(np.amin(V)))
    # print("R max = " + str(np.amax(R)) + ", R min = " + str(np.amin(R)))
    # print("G max = " + str(np.amax(G)) + ", G min = " + str(np.amin(G)))
    # print("B max = " + str(np.amax(B)) + ", B min = " + str(np.amin(B)))

    return rgb_image

def rgb2yuv2(image, bitDepth):

    maxValue = float(2**bitDepth)

    R = np.float32(image[:,:,0])
    G = np.float32(image[:,:,1])
    B = np.float32(image[:,:,2])

    Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
    V =  (maxValue/2.) - (0.168736 * R) - (0.331264 * G) + (0.5 * B)
    U =  (maxValue/2.) + (0.5 * R) - (0.418688 * G) - (0.081312 * B)

    yuv_image = np.array([Y,U,V])
    yuv_image = np.swapaxes(yuv_image,0,2)
    yuv_image = np.swapaxes(yuv_image,0,1)

    return yuv_image

def separate_frames(hr_imgs, lr_imgs):

    numSeqs = len(hr_imgs)
    if len(lr_imgs) != numSeqs:
        raise Exception("Lists do not contain the same number of elements")

    numFrames = len(hr_imgs[0])

    hr_imgs_new = []
    lr_imgs_new = []

    for i in range(0, numSeqs):
        
        frameListHR = hr_imgs[i]
        frameListLR = lr_imgs[i]
        
        for j in range(0,numFrames):

            hr_imgs_new.append(frameListHR[j])
            lr_imgs_new.append(frameListLR[j])
    
    return hr_imgs_new, lr_imgs_new

def unison_shuffle(*tensors):
	
    c = list(zip(*tensors))
    random.seed(10)
    random.shuffle(c)
    out = zip(*c)

    return out

def normalize_array(x, bitDepth):
    maxValue = float(2**bitDepth - 1) 
    x = x / (maxValue)
    #x = x / (maxValue / 2.)
    #x = x - 1.
    return x

def inverse_normalize_array(x, bitDepth):
    maxValue = float(2**bitDepth - 1) 
    #x = x + 1.
    #x = x * (maxValue / 2.)
    x = x * maxValue
    return x

def check_HR_LR_match(HR_list, LR_list):

    if len(HR_list) != len(LR_list):
        raise Exception("Lists do not contain the same number of elements")

    for i in range(0,len(HR_list)):

        myParams_HR = videoParams()
        myParams_HR.filename = HR_list[i]
        myParams_HR.extractVideoParameters()

        myParams_LR = videoParams()
        myParams_LR.filename = LR_list[i]
        myParams_LR.extractVideoParameters()

        if myParams_HR.seqName != myParams_LR.seqName:
            print("myParams_HR.seqName = " + myParams_HR.seqName + " | myParams_LR.seqName = " + myParams_LR.seqName)
            raise Exception("Elements from HR list and LR list do not match")

def print2logFile(filename, text, first=0):

    if first==1:
        logFile = open(filename,'w')
    else: 
        logFile = open(filename, 'a')

    logFile.write(text)

    logFile.close()

def calculate_mse(image1, image2):

    se = np.power(image1 - image2, 2)
    mse = np.mean(se)

    return mse

def convertInt32(array, bitDepth):

    maxValue = ((2 ** bitDepth) - 1)
    array[array < 0] = 0
    array[array > maxValue] = maxValue

    array = np.int32(array)

    return array

def convertInt16(array, bitDepth):

    maxValue = ((2 ** bitDepth) - 1)
    array[array < 0] = 0
    array[array > maxValue] = maxValue

    array = np.int16(array)

    return array

def padarray(im, pad_size):
    im = np.array(im)

    h_pad = pad_size[0]
    w_pad = pad_size[1]

    size = np.shape(im)
    h = size[0]
    w = size[1]

    if (im.ndim == 3):
        d = size[2]

        im_pad = np.zeros((h + 2 * h_pad, w + 2 * w_pad, d))
        im_pad[h_pad:h + h_pad, w_pad:w + w_pad, :] = im

        for k in range(0, d):

            if h_pad != 0:
                for j in range(w_pad, w + w_pad):
                    im_pad[0:h_pad, j, k] = im[h_pad - 1::-1, j - w_pad, k]

                    im_pad[h + h_pad:, j, k] = im[h - 1:h - h_pad - 1:-1, j - w_pad, k]

            if w_pad != 0:
                for i in range(0, h + 2 * h_pad):
                    im_pad[i, 0:w_pad, k] = im_pad[i, w_pad * 2 - 1:w_pad - 1:-1, k]

                    im_pad[i, w + w_pad:, k] = im_pad[i, w + w_pad - 1:w - 1:-1, k]

    else:

        im_pad = np.zeros((h + 2 * h_pad, w + 2 * w_pad))
        im_pad[h_pad:h + h_pad, w_pad:w + w_pad] = im

        if h_pad != 0:
            for j in range(w_pad, w + w_pad):
                im_pad[0:h_pad, j] = im[h_pad - 1::-1, j - w_pad]

                im_pad[h + h_pad:, j] = im[h - 1:h - h_pad - 1:-1, j - w_pad]

        if w_pad != 0:
            for i in range(0, h + 2 * h_pad):
                im_pad[i, 0:w_pad] = im_pad[i, w_pad * 2 - 1:w_pad - 1:-1]

                im_pad[i, w + w_pad:] = im_pad[i, w + w_pad - 1:w - 1:-1]

    return im_pad


def imresize_L3(im_lr, scale=2):
    # only works for scale=2 until now

    kernel_size = 6

    x_even = kernel_size / scale + 0.5 * (1 - 1 / scale) - np.array([1, 2, 3, 4, 5, 6])
    x_odd = (kernel_size + 1) / scale + 0.5 * (1 - 1 / scale) - np.array([1, 2, 3, 4, 5, 6])

    W_even = np.zeros(x_even.size)
    W_odd = np.zeros(x_even.size)
    for i in range(0, x_even.size):
        W_even[i] = (math.sin(math.pi * x_even[i]) * math.sin(math.pi * (x_even[i] / 3))) / (
                    np.power(math.pi, 2) * np.power(x_even[i], 2) / 3)
        W_odd[i] = (math.sin(math.pi * x_odd[i]) * math.sin(math.pi * (x_odd[i] / 3))) / (
                    np.power(math.pi, 2) * np.power(x_odd[i], 2) / 3)

    im_lr_pad = padarray(im_lr, np.array([0, 3]))

    size = np.shape(im_lr)
    h = size[0]
    w = size[1]

    if (im_lr.ndim == 3):

        d = size[2]

        im_hr_h = np.zeros([h, scale * w, d])

        im_hr = np.zeros([scale * h, scale * w, d])

        for k in range(0, d):

            for i in range(0, h):
                i_hr = 0

                for j in range(0, w + 1):

                    pixels_lr = im_lr_pad[i, j:j + kernel_size, k]

                    if j != w:
                        im_hr_h[i, i_hr, k] = np.matmul(pixels_lr, W_even.transpose())
                        i_hr += 1

                    if j != 0:
                        im_hr_h[i, i_hr, k] = np.matmul(pixels_lr, W_odd.transpose())
                        i_hr += 1

            im_hr_h_pad = padarray(im_hr_h, np.array([3, 0]))

            for i in range(0, scale * w):
                i_hr = 0

                for j in range(0, h + 1):

                    pixels_lr = im_hr_h_pad[j:j + kernel_size, i, k]

                    if j != h:
                        im_hr[i_hr, i, k] = np.matmul(pixels_lr, W_even.transpose())
                        i_hr += 1

                    if j != 0:
                        im_hr[i_hr, i, k] = np.matmul(pixels_lr, W_odd.transpose())
                        i_hr += 1

    else:

        im_hr_h = np.zeros([h, scale * w])

        for i in range(0, h):
            i_hr = 0

            for j in range(0, w + 1):

                pixels_lr = im_lr_pad[i, j:j + kernel_size]

                if j != w:
                    im_hr_h[i, i_hr] = np.matmul(pixels_lr, W_even.transpose())
                    i_hr += 1

                if j != 0:
                    im_hr_h[i, i_hr] = np.matmul(pixels_lr, W_odd.transpose())
                    i_hr += 1

        im_hr_h_pad = padarray(im_hr_h, np.array([3, 0]))

        im_hr = np.zeros([scale * h, scale * w])

        for i in range(0, scale * w):
            i_hr = 0

            for j in range(0, h + 1):

                pixels_lr = im_hr_h_pad[j:j + kernel_size, i]

                if j != h:
                    im_hr[i_hr, i] = np.matmul(pixels_lr, W_even.transpose())
                    i_hr += 1

                if j != 0:
                    im_hr[i_hr, i] = np.matmul(pixels_lr, W_odd.transpose())
                    i_hr += 1

    return im_hr


def calculate_mean_block(hr_list, hr_path, lr_list, lr_path, batch_size, ratio, block_size, save_folder ):

    # check if mean_block has been computed before:

    mean_block_HR, mean_block_LR, foundFlag = restore_mean_block(save_folder, ratio, block_size) # check if mean blocks were saved previously

    if not foundFlag:

        # initialize sum of blocks for mean block calculation
        mean_block_HR = np.zeros((int(ratio * block_size), int(ratio * block_size), 3))
        mean_block_LR = np.zeros((block_size, block_size, 3))
        block_count = 0

        for idx in range(0, len(hr_list), batch_size):

            imgs_HR = np.asarray(read_blocks(hr_list[idx : idx + batch_size], path=hr_path, mode='HR', ratio=ratio, block_size=block_size, n_threads=16, dimensions=3))
            imgs_LR = np.asarray(read_blocks(lr_list[idx: idx + batch_size], path=lr_path, mode='LR', ratio=ratio,  block_size=block_size, n_threads=16, dimensions=3))

            for block in range(0, batch_size):

                mean_block_HR = mean_block_HR + imgs_HR[block]
                mean_block_LR = mean_block_LR + imgs_LR[block]
                block_count += 1

        # Calculate and save mean_block_HR and mean_block_LR

        mean_block_HR = mean_block_HR / block_count
        outfilename = os.path.join(save_folder, 'mean_block_HR.png')
        rgb_block = yuv2rgb(255*mean_block_HR, 8)
        imageio.imwrite(outfilename, np.uint8(rgb_block))
        mean_block_HR_10bit = inverse_normalize_array(mean_block_HR,10) # here I am using 10 bit as the maximum not to loose information (10 bit or 8 bit inputs)
        save_mean_block(convertInt32(mean_block_HR_10bit,10), 10, save_folder, 'HR')

        mean_block_LR = mean_block_LR / block_count
        outfilename = os.path.join(save_folder, 'mean_block_LR.png')
        rgb_block = yuv2rgb(255*mean_block_LR, 8)
        imageio.imwrite(outfilename, np.uint8(rgb_block))
        mean_block_LR_10bit = inverse_normalize_array(mean_block_LR, 10) # here I am using 10 bit as the maximum not to loose information (10 bit or 8 bit inputs)
        save_mean_block(convertInt32(mean_block_LR_10bit,10), 10, save_folder, 'LR')

    return mean_block_HR, mean_block_LR


def restore_mean_block(save_folder, ratio, block_size):

    mean_block_HR = []
    mean_block_LR = []

    if (os.path.isfile(os.path.join(save_folder, 'mean_block_HR.image')) == True) and (os.path.isfile(os.path.join(save_folder, 'mean_block_LR.image')) == True):
        mean_block_HR = get_blocks_fn('mean_block_HR.image', save_folder, 'HR', ratio, block_size, 3)
        mean_block_LR = get_blocks_fn('mean_block_LR.image', save_folder, 'LR', ratio, block_size, 3)
        return mean_block_HR, mean_block_LR, True
    else:
        return mean_block_HR, mean_block_LR, False

def chromaSub_420_multiple(yuv_image):

    y_im = yuv_image[:, :, :, 0]
    u_im = np.repeat(np.repeat(yuv_image[:, 0::2, 0::2, 1], 2, axis=1), 2, axis=2)
    v_im = np.repeat(np.repeat(yuv_image[:, 0::2, 0::2, 2], 2, axis=1), 2, axis=2)

    yuv_im = np.stack((y_im, u_im, v_im), axis=3)

    return yuv_im

def resize_subsample_420(yuv_image, size):

    imY_up = resize(yuv_image[:,:,0], size, order=3, mode='reflect')

    imU_up = np.repeat(np.repeat(resize(yuv_image[0::2,0::2,1], size*0.5, order=3, mode='reflect'), 2, axis=0), 2, axis=1)

    imV_up = np.repeat(np.repeat(resize(yuv_image[0::2,0::2,2], size*0.5, order=3, mode='reflect'), 2, axis=0), 2, axis=1)

    im_up = np.stack((imY_up, imU_up, imV_up), axis=2)

    return im_up

def resize_single(im, size=[100, 100], interpUV="nearest", format="YUV420"):
    """Resize an image by given output size and method. Warning, this function
    will rescale the value to [0, 255]. """

    if format == "YUV420":

        size = np.asarray(size)
        imY = im[:,:,0]
        imY_up = resize(imY, size, order=3, mode='reflect')

        imU = im[:,:,1]
        if interpUV == "nearest":
            imU_up = resize(imU, size, order=0, mode='reflect')
        else:
            imU = imU[0::2,0::2]
            imU_up = np.repeat(np.repeat(resize(imU, size*0.5, order=3, mode='reflect'), 2, axis=0), 2, axis=1)

        imV = im[:,:,2]
        if interpUV == "nearest":
            imV_up = resize(imV, size, order=0, mode='reflect')
        else:
            imV = imV[0::2,0::2]
            imV_up = np.repeat(np.repeat(resize(imV, size*0.5, order=3, mode='reflect'), 2, axis=0), 2, axis=1)

        im_up = np.stack((imY_up, imU_up, imV_up), axis=2)

    else:
        im_up = resize(im, size, order=3, mode='reflect')

    return np.asarray(im_up)

def resize_multiple(x, size=[100, 100], interpUV="nearest", format="YUV420"):
    """Resize an image by given output size and method. Warning, this function
    will rescale the value to [0, 255].

    Parameters
    -----------
    x : Tupple of numpy arrays - several images (Nimages, row, col, channel)
        An image with dimension of [row, col, channel] (default).
    size : int, float or tuple (h, w)
        - int, Percentage of current size.
        - float, Fraction of current size.
        - tuple, Size of the output image.
    interp : str, optional
        Interpolation to use for re-sizing (‘nearest’, ‘lanczos’, ‘bilinear’, ‘bicubic’ or ‘cubic’).
    mode : str, optional
        The PIL image mode (‘P’, ‘L’, etc.) to convert arr before resizing.

    Returns
    --------
    imresize : ndarray
    The resized array of image.

    References
    ------------
    - `scipy.misc.imresize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html>`_
    """

    array_up = []
    for idxIm in range(len(x)):
        im = np.squeeze(x[idxIm])

        if format == "YUV420":

            size = np.asarray(size)
            imY = im[:,:,0]
            imY_up = resize(imY, size, order=3, mode='reflect')

            imU = im[:,:,1]
            if interpUV == "nearest":
                imU_up = resize(imU, size, order=0, mode='reflect')
            else:
                imU = imU[0::2,0::2]
                imU_up = np.repeat(np.repeat(resize(imU, size*0.5, order=3, mode='reflect'), 2, axis=0), 2, axis=1)

            imV = im[:,:,2]
            if interpUV == "nearest":
                imV_up = resize(imV, size, order=0, mode='reflect')
            else:
                imV = imV[0::2,0::2]
                imV_up = np.repeat(np.repeat(resize(imV, size*0.5, order=3, mode='reflect'), 2, axis=0), 2, axis=1)

            im_up = np.stack((imY_up, imU_up, imV_up), axis=2)

        else:
            im_up = resize(im, size, order=3, mode='reflect')

        array_up.append(im_up)

    return np.asarray(array_up)

'''
Code below is adapted from https://github.com/mtyka/laploss
'''

def gauss_kernel(size=5, sigma=1.0):
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    return kernel

def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
    t_kernel = tf.reshape(tf.constant(gauss_kernel(size=k_size, sigma=sigma), tf.float32),
                            [k_size, k_size, 1, 1])
    t_kernel3 = tf.concat([t_kernel]*t_input.get_shape()[3], axis=2)
    t_result = t_input
    for r in range(repeats):
        t_result = tf.nn.depthwise_conv2d(t_result, t_kernel3,
            strides=[1, stride, stride, 1], padding='SAME')
    return t_result

def make_laplacian_pyramid(t_img, max_levels):
    t_pyr = []
    current = t_img
    for level in range(max_levels):
        t_gauss = conv_gauss(current, stride=1, k_size=5, sigma=1.0)
        t_diff = current - t_gauss
        t_pyr.append(t_diff)
        current = tf.nn.avg_pool(t_gauss, [1,2,2,1], [1,2,2,1], 'VALID')
    t_pyr.append(current)
    return t_pyr

def laploss(t_img1, t_img2, max_levels=4):
    """
    inputs: (B,H,W,C)
    """
    t_pyr1 = make_laplacian_pyramid(t_img1, max_levels)
    t_pyr2 = make_laplacian_pyramid(t_img2, max_levels)
    t_losses = [tf.norm(a-b,ord=1)/tf.size(a, out_type=tf.float32) for a,b in zip(t_pyr1, t_pyr2)]
    t_loss = tf.reduce_sum(t_losses)*tf.shape(t_img1, out_type=tf.float32)[0]
    return t_loss