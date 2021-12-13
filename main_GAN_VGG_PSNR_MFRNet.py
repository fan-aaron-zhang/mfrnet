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

# ====================== HYPER-PARAMETERS ===========================###
# Adam
beta1 = 0.9
lr_decay = 0.1
vgg_conNum = 6

# evaluation
SIZE_SUBBLOCK = 168
SIZE_OVERLAP = 4



def evaluate(afterTrain=False):
    valid_lr_img_path = tl.global_flag['evalLR']
    valid_hr_img_path = tl.global_flag['evalHR']
    testModel = tl.global_flag['testModel']

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
        batch_size = 16
        generator = ESRGAN_OutCascading_InShareSkip_Rethinking_efficient_g
        CNN_name = 'BD9bit_YUV'
        valid_block_size = 96
    else:
        batch_size = 16
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

    ni = int(np.sqrt(batch_size))

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

                maxValue = pow(2, currVideo.bitDepth) - 1

                iframe = 0

                valid_lr_img = np.squeeze(
                    loadYUVfile(os.path.join(valid_lr_img_path, currVideo.filename), currVideo.width, currVideo.height,
                                np.array([iframe]), currVideo.colorSampling, currVideo.bitDepth))


                size_LR = valid_lr_img.shape
                size_HR = valid_lr_img.shape

                if splitBlocks == 1:

                    SIZE_SUBBLOCK = math.gcd(size_LR[0], size_LR[1]) + SIZE_OVERLAP * 2


                    t_image = tf.placeholder('float32', [None, SIZE_SUBBLOCK, SIZE_SUBBLOCK, input_depth],
                                             name='input_image')  # the 1 in the last dimension is because we are just using the Y channel

                    t_image_up = tf.placeholder('float32',
                                                [None, int(SIZE_SUBBLOCK * ratio), int(SIZE_SUBBLOCK * ratio),
                                                 input_depth], name='input_image_up')

                else:
                    t_image = tf.placeholder('float32', [None, size_LR[0], size_LR[1], input_depth],
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
                sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
                tl.layers.initialize_global_variables(sess)

                if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_model,
                                                network=net_g) is False:
                    raise Exception("Error loading trained model.")

                ##========================== Get mean frame =========================###

                if input_depth == 1:
                    mean_image_LR = mean_dims_LR[0] * np.ones((size_LR[0], size_LR[1], 1))
                    # mean_image_HR = mean_dims_HR[0] * np.ones((1, size_HR[0], size_HR[1], 1))
                else:
                    mean_image_Y = mean_dims_LR[0] * np.ones((size_LR[0], size_LR[1], 1))
                    mean_image_U = mean_dims_LR[1] * np.ones((size_LR[0], size_LR[1], 1))
                    mean_image_V = mean_dims_LR[2] * np.ones((size_LR[0], size_LR[1], 1))
                    mean_image_LR = np.concatenate((mean_image_Y, mean_image_U, mean_image_V), axis=2)


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

                while valid_lr_img.ndim != 0 and iframe < nframes:

                    print("[*] upsampling sequence " + video_file + " frame " + str(iframe + 1))
                    start_time = time.time()

                    if input_depth == 1:
                        valid_lr_img_input = normalize_array(valid_lr_img[:, :, 0],
                                                             currVideo.bitDepth)  # keep only the Y channel
                        valid_lr_img_input = np.expand_dims(valid_lr_img_input, axis=2)
                    else:
                        valid_lr_img_input = normalize_array(valid_lr_img, currVideo.bitDepth)

                    # subtract mean
                    if ratio != 1:
                        valid_lr_img_input = valid_lr_img_input * 2 - mean_image_LR


                    genFrame = np.zeros((1, size_HR[0], size_HR[1], input_depth))
                    count_blocks = 0

                    for i_w in range(0, w_blocks):
                        for i_h in range(0, h_blocks):

                            valid_lr_img_block = valid_lr_img_input[height_start[i_h]:height_end[i_h],
                                                 width_start[i_w]:width_end[i_w], :]

                            size_valid_block = np.shape(valid_lr_img_block)

                            valid_lr_img_block_up = resize_single(valid_lr_img_block,
                                                                  size=[int(ratio * size_valid_block[0]),
                                                                        int(ratio * size_valid_block[1])],
                                                                  interpUV=interpUV, format="YUV420")


                            ###======================= EVALUATION =============================###

                            genBlock = sess.run(net_g.outputs,
                                                {t_image: [valid_lr_img_block], t_image_up: [valid_lr_img_block_up]})

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


                    print("[*] upsampling took: %4.4fs, LR size: %s /  generated HR size: %s" % (
                        time.time() - start_time, valid_lr_img_input.shape, genFrame.shape))

                    if ratio != 1:
                        genFrame = (genFrame + mean_image_HR) / 2  # add mean that was previously subtracted

                    if iframe == 0:
                        mode = 'wb'
                    else:
                        mode = 'ab'

                    size_HR = genFrame[0].shape

                    genFrame = inverse_normalize_array(genFrame[0], currVideo.bitDepth)


                    if input_depth == 1 and ratio == 1:
                        genFrameYUV = np.stack((genFrame[:, :, 0], valid_lr_img[:, :, 1], valid_lr_img[:, :, 2]),
                                               axis=2)
                    else:
                        # if input_depth == 3:
                        #    genFrameYUV = rgb2yuv(genFrame, currVideo_hr.bitDepth)
                        # else:
                        genFrameYUV = genFrame


                    if genFrameYUV.shape[2] == 3:
                        genFrameYUV = np.resize(genFrameYUV, (1, size_HR[0], size_HR[1], 3))
                        colorSave = currVideo.colorSampling
                    else:
                        genFrameYUV = np.resize(genFrameYUV, (1, size_HR[0], size_HR[1], 1))
                        colorSave = 'Y'


                    print("[*] saving frame")

                    saveYUVfile(convertInt32(genFrameYUV, currVideo.bitDepth),
                                save_dir, mode, colorSave,
                                currVideo.bitDepth)

                    iframe += 1
                    frame_count += 1

                    valid_lr_img = np.squeeze(
                        loadYUVfile(os.path.join(valid_lr_img_path, currVideo.filename), currVideo.width,
                                    currVideo.height,
                                    np.array([iframe]), currVideo.colorSampling, currVideo.bitDepth))

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

    if tl.global_flag['mode'] == 'evaluate':
        evaluate(afterTrain=False)
    else:
        raise Exception("Unknow --mode")
