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

def get_blocks_batch_fn(filename, path, HRorLR, ratio, block_size, dimensions, batch_size):

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

    multiplier =  int(sizeBytes / (batch_size*sizeFrame*3))

    if multiplier == 1:
        bitDepth = 8
        elementType = 'B'
    elif multiplier == 2:
        bitDepth = 10
        elementType = 'H'
    else:
        raise Exception("Error reading file: multiplier should be either 1 or 2")

    all_blocks = []
    with open(os.path.join(path,filename),'rb') as fileID:

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

        # if HRorLR == 'HR':
        #     block = denoise_image(block)

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

def unison_shuffle(a, b):
	
    c = list(zip(a, b))
    random.seed(10)
    random.shuffle(c)
    a, b = zip(*c)

    return a, b

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