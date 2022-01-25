import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle
import imgaug as ia
from imgaug import augmenters as iaa

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def noisy(image, noise_typ):
    if noise_typ == "gaussian":
        # np.array([103.939, 116.779, 123.68])
        mean = 0
        var = 0.01*255.0
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,image.shape)
        gauss = gauss.reshape(image.shape)
        noisy = image + gauss
        return noisy
    elif noise_typ == "salt&pepper":
        s_vs_p = 0.5
        amount = 0.05
        out = image.copy()
        
        # print image.size
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, np.max((i-1,1)), int(num_salt)) for i in image.shape]
        out[coords] = 255.0

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, np.max((i-1,1)), int(num_pepper)) for i in image.shape]
        out[coords] = 0.0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(np.abs(image) * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        gauss = np.random.randn(*image.shape)
        gauss = gauss.reshape(image.shape)        
        noisy = image + image * 0.1* gauss
        return noisy
    else:
        return image.copy()

def imageAugmentation(inputImages):
    noiseTypeList = ['gaussian', 'salt&pepper', 'poisson', 'speckle']
    random.shuffle(noiseTypeList)
    select = np.random.randint(0, 2, len(noiseTypeList))
    for i in range(len(noiseTypeList)):
        if select[i] == 1:
            inputImages = noisy(image=inputImages, noise_typ=noiseTypeList[i])
    return inputImages

'''https://github.com/aleju/imgaug'''
def imgAug(inputImage, crop=True, flip=True, gaussianBlur=True, channelInvert=True, brightness=True, hueSat=True):
    augList = []
    if crop:
        augList += [iaa.Crop(px=(0, 16))] # crop images from each side by 0 to 16px (randomly chosen)
    if flip:
        augList += [iaa.Fliplr(0.5)] # horizontally flip 50% of the images
    if gaussianBlur:
        augList += [iaa.GaussianBlur(sigma=(0, 3.0))] # blur images with a sigma of 0 to 3.0
    if channelInvert:
        augList += [iaa.Invert(0.05, per_channel=True)] # invert color channels
    if brightness:
        augList += [iaa.Add((-10, 10), per_channel=0.5)] # change brightness of images (by -10 to 10 of original value)
    if hueSat:
        augList += [iaa.AddToHueAndSaturation((-20, 20))] # change hue and saturation
    seq = iaa.Sequential(augList)
    # seq = iaa.Sequential([
    #     iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    #     # iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    #     iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
    #     iaa.Invert(0.05, per_channel=True),  # invert color channels
    #     iaa.Add((-10, 10), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
    #     iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
    # ])

    image_aug = seq.augment_image(inputImage)
    return image_aug

def imageRandomAugmentation(
        inputImage, imageRowFinal, imageColFinal, padding=True,
        imAug=True, imAugPr = 0.5,
        randomTrans=True, randomScale=True, transPr=0.5, scalePr=0.5,
        transRatioMax=0.2, scaleRatioMin=0.8, scaleRatioMax=1.2):
    imageRow, imageCol, _ = inputImage.shape
    dRow, dCol = 0, 0
    if imAug:
        if np.random.rand()<imAugPr:
            gaussian, brightness, hueSat, channelInvert = np.random.choice(a=[False, True], size=4, p=[0.5, 0.5])
            inputImage = imgAug(inputImage=inputImage, crop=False, flip=False,
                                              gaussianBlur=gaussian, channelInvert=channelInvert,
                                              brightness=brightness, hueSat=hueSat)

    rowPadding, colPadding = 0, 0
    if padding:
        expectedRow, expectedCol = imageRowFinal, imageColFinal
        if imageRow > expectedRow:
            eRowTemp, eColTemp = expectedRow, expectedCol
            expectedRow = eRowTemp * float(imageRow)/float(eRowTemp)
            expectedCol = eColTemp * float(imageRow)/float(eRowTemp)
        if imageCol > expectedCol:
            eRowTemp, eColTemp = expectedRow, expectedCol
            expectedRow = eRowTemp * float(imageCol)/float(eColTemp)
            expectedCol = eColTemp * float(imageCol)/float(eColTemp)
        rowGap, colGap = expectedRow - imageRow, expectedCol - imageCol

        if rowGap*expectedCol < colGap*expectedRow:
            rowPadding = 0
            colPadding = int((colGap - rowGap*float(expectedCol)/float(expectedRow))/2)
        elif rowGap*expectedCol > colGap*expectedRow:
            rowPadding = int((rowGap - colGap*float(expectedRow)/float(expectedCol))/2)
            colPadding = 0
        inputImage = cv2.copyMakeBorder(
            src=inputImage, top=rowPadding, bottom=rowPadding,
            left=colPadding, right=colPadding, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    imageRow, imageCol, _ = inputImage.shape

    dRowScale, dColScale = 0,0
    scaleRatio = 1.0
    if randomScale:
        if np.random.rand() < scalePr:
            scaleRatio = np.random.uniform(low=scaleRatioMin, high=scaleRatioMax)
    # dRowScale = (imageRow - scaleRatio*imageRow)/2.0
    # dColScale = (imageCol - scaleRatio*imageCol)/2.0
    scaleMat = scaleRatio*np.float32([[1.0, 0, -imageCol/2], [0, 1.0, -imageRow/2]]) + np.float32([[0,0,imageCol/2],[0,0,imageRow/2]])

    dRowTrans, dColTrans = 0,0
    if randomTrans:
        if np.random.rand() < transPr:
            rowTransMax, colTransMax = int(imageRow*transRatioMax), int(imageCol*transRatioMax)
            dRowTrans, dColTrans = np.random.randint(-rowTransMax,rowTransMax), np.random.randint(-colTransMax,colTransMax)
    transMat = np.float32([[0,0,dColTrans], [0,0,dRowTrans]])
    affineMat = scaleMat + transMat

    inputImage = cv2.warpAffine(inputImage, affineMat, (imageCol, imageRow))
    inputImage = cv2.resize(inputImage, (imageColFinal, imageRowFinal), interpolation=cv2.INTER_CUBIC)
    # dRow = rowPadding + dRowTrans
    # dCol = colPadding + dColTrans

    return inputImage, rowPadding, dRowTrans, colPadding, dColTrans, scaleRatio

