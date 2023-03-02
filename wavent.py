import math
import numpy as np
import cv2
import pywt
from collections import Counter

def entropy(coE):
    p, lns = Counter(coE), float(len(coE))
    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())
 

def hsvWeight(img, mxLc):
    '''made by cheol woo park'''
    # Get the average value of a part.
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgHsv = im2double(imgHsv)
    dataV = []
    
    for i in range(0, len(mxLc)):
        tempIm = imgHsv[:,:,2]
        tempIm = tempIm[mxLc[i]]
        dataV = np.hstack((dataV, np.mean(tempIm))) 
    return dataV

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def waveEnt(img, mxLc):
    ''' made by cheol woo park'''
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.dwt2(imgG, 'db2')
    LL, (LH, HL, HH) = coeffs
    
    tempCo = np.zeros(shape=(np.size(LL,0)-2,np.size(LL,1)-2), dtype=np.float64)
    # wavelet leader
    tempCo = cv2.max(cv2.max(HL,LH),HH)
    tempCo = tempCo[1:np.size(tempCo,0)-1,1:np.size(tempCo,1)-1]
    # zero padding
    imgStd = np.std(tempCo)
    tempCo[abs(tempCo) < imgStd] = 0
    
    entCal = [0 for _ in range(len(mxLc))]
    zeroCal = [0 for _ in range(len(mxLc))]
    dataCal = [0 for _ in range(len(mxLc))]
    for i in range(0, len(mxLc)):
        tempLc = mxLc[i][::2,::2]
        tempLc = tempLc[0:np.size(tempCo,0),0:np.size(tempCo,1)]
        entCal[i] = entropy(tempCo[tempLc]) 
        # zero rate
        zeroRate = tempCo[tempLc==1]
        zeroCal[i] = sum(zeroRate==0)/sum(sum(tempLc))
        
    temp = sum(entCal)
    if len(entCal) != 1:
        for i in range(0, len(entCal)):
            entCal[i] = 1 - entCal[i]/temp
            dataCal[i] = entCal[i] * zeroCal[i]
    else:
        entCal[0] = 1
        dataCal[i] = entCal[i] * zeroCal[i]
    return dataCal

def mapRiver(img, mxIm, mxLc, dataCal, dataV):
    ''' made by cheol woo park'''
    dumCal = np.zeros([img.shape[0], img.shape[1]], dtype='float64')
    dumLg = np.zeros([img.shape[0], img.shape[1]], dtype=bool)
    riverP = [0 for _ in range(len(dataCal))]
    for i in range(0, len(dataCal)):
        # If value is more than 0.7, it is judged not to be river.
        if dataV[i] >=  0.7:
            riverP[i] = 0
        else:
            riverP[i] = dataCal[i]
            
    temp = max(riverP)
    if temp != 0:
        riverP = riverP / temp
        
        for i in range(0, len(riverP)):
            if riverP[i] >= 0.95:
                dumLg = dumLg + mxLc[i]
        
        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgHsv = im2double(imgHsv)
        img2 = im2double(img)
        dumImg = img2 * 0
        
        img2[:,:,0] = img2[:,:,0] * dumLg
        img2[:,:,1] = img2[:,:,1] * dumLg
        img2[:,:,2] = img2[:,:,2] * dumLg
            
        tempImH = imgHsv[:,:,0]
        tempImH[tempImH > 5/6] = tempImH[tempImH > 5/6] - 1
        tempImH = 1 - 2*abs(tempImH-1/3)
        for i in range(0, np.size(tempImH, 0)):
            for j in range(0, np.size(tempImH, 1)):
                    tempImH[i,j] = (math.erf(6*tempImH[i,j]-3)+1)/2   
        tempImH = tempImH * dumLg
        
        # Removed top 1% saturation
        tempImS = imgHsv[:,:,1]
        tempImS0 = np.sort(tempImS[dumLg == 1])
        tempS0 = tempImS0[round(np.size(tempImS0)*0.99)]
        tempImSp = tempImS / tempS0
        tempImSp[tempImSp > 1] = 0
        
        # mapping to image
        rg = (img2[:,:,1]+img2[:,:,0])
        rgTemp = rg
        rg[rg == 0] = 1
        dumCal = img2[:,:,1] / rg * tempImH * tempImSp
        dumCal[rgTemp == 0] = 0
        dumCal = 12*(dumCal-1/2)
        dumCal = 1/(1+np.exp(-dumCal))        
        dumCal[dumCal>1] = 1  
        dumCal[dumCal<0] = 0  
        dumCal = dumCal * dumLg
        
        dumImg = np.ones(shape=(np.size(dumCal,0),np.size(dumCal,1),3), dtype=np.float64)
        dumImg[:,:,0] = dumImg[:,:,0] * dumLg
        dumImg[:,:,1] = dumImg[:,:,1] * dumLg
        dumImg[:,:,0] = dumImg[:,:,0] - dumCal
        dumImg[:,:,2] = dumImg[:,:,0]
        dumImg = dumImg * 255
        dumImg = dumImg.astype(np.uint8)
        
        imgOut = img.astype(np.uint8)
        imgOut[:,:,0] = imgOut[:,:,0] * ~dumLg
        imgOut[:,:,1] = imgOut[:,:,1] * ~dumLg
        imgOut[:,:,2] = imgOut[:,:,2] * ~dumLg
        
        imgRiver = imgOut+dumImg
    else:
        imgRiver = img * 0
    
    return imgRiver
