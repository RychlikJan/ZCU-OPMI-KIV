import sys
import os
from collections.abc import Iterable, Mapping
import mne
from mne.datasets import sample
import matplotlib.pyplot as plt
import numpy as np
import math
import glob, os
from scipy.fft import fft, ifft
import pandas as pd


valuesPZAll= np.ones((20,16))
heartBeat =np.ones((20,9))
succesfull=np.ones((20,8))
shownCards=np.ones((20,8))

manIndex = np.array([0.,1.,5.,7.,13.,14.,15.,17.,18.,19.])

excel = pd.read_excel("D:/ZCU/5.rocnik/OborovyProjekt/DP Klára Beránková/Výsledné hodnoty měřených dat/Vysledky_puls_a_hra.xlsx")
excelNP = np.array(excel)
def WindowSlide(arr, k):
    n = len(arr)
    nk = n-k
    if n < k:
        print("Invalid window slide input")
        return -1

    window_sum = sum(arr[:k])
    windowSlideResults = [0] * (nk+1);

    for i in range(nk):
        windowSlideResults[i] = window_sum/(nk)
        if(window_sum < -90):
            print(window_sum)
        window_sum = window_sum - arr[i] + arr[i + k]
        windowSlideResults[i+1] = window_sum / (nk)

    windowSlideResults= fft(windowSlideResults)
    return windowSlideResults


def recountValues(arr):
    returnArr= np.empty(arr[0].size)
    for i in range(arr[0].size):
        returnArr[i]= -(10* (np.log10(arr[0,i])))
    return returnArr


def showData(filePath, pos,pos2):
    mneraw = mne.io.read_raw_brainvision(filePath,preload=True)
    #mneraw = mne.io.read_raw_brainvision("D:/ZCU/5.rocnik/OborovyProjekt/DP Klára Beránková/EEG data/NeupravenaEEGdata/Subjekt 6/Pexeso_berankova0041.vhdr",preload=True)

    names = mneraw.ch_names
    print('The first few channel names are {}.'.format(', '.join(names[:])))

    sampling_freq = mneraw.info['sfreq']
    mneraw = mneraw.filter(0.1, 30., fir_design='firwin')

    mneraw = mneraw.del_proj()
    if pos ==0:
        mneraw.plot_psd(fmin= 0.1,fmax=30.)


    #### fot to chanel index
    channel_index = 2
    start_stop_secondsAlpha = np.array([8., 12.]) #filter hranice
    start_sampleAlpha, stop_sampleAlpha = (start_stop_secondsAlpha * sampling_freq).astype(int)
    raw_selectionAlpha = mneraw[channel_index, start_sampleAlpha:stop_sampleAlpha]

    start_stop_secondsBeta = np.array([18., 30.])  # filter hranice
    start_sampleBeta, stop_sampleBeta = (start_stop_secondsBeta * sampling_freq).astype(int)
    raw_selectionBeta = mneraw[channel_index, start_sampleBeta:stop_sampleBeta]

    print("Raw")
    x = raw_selectionAlpha[1]
    y = recountValues(raw_selectionAlpha[0]).T
    pomAlpha = 0;
    for i in y:
        if not(math.isnan(i)):
            pomAlpha = pomAlpha +i
    print("Alpha: "+ str(pomAlpha/y.__len__()))
    print(pos2,pos)
    valuesPZAll[pos2,pos] = pomAlpha/y.__len__()

    x = raw_selectionBeta[1]
    y = recountValues(raw_selectionBeta[0]).T
    pomBeta = 0;
    for i in y:
        if not(math.isnan(i)):
            pomBeta = pomBeta +i
    print("Beta: "+str(pomBeta/y.__len__()))
    valuesPZAll[pos2,pos+1] = pomBeta / y.__len__()


def reworkExcel():
    position = 0
    for i in range(0,58,3):
        arrayPom = excelNP[i]
        for j in range(1,arrayPom.__len__()):
            shownCards[position,j-1] = arrayPom[j]
        arrayPom = excelNP[i+1]
        for j in range(1,arrayPom.__len__()):
            succesfull[position,j-1] = arrayPom[j]
        arrayPom = excelNP[i+2]
        for j in range(arrayPom.__len__()):
            heartBeat[position,j] = arrayPom[j]
        position = position +1

def means():
    subjectMean = np.mean(shownCards, axis=1)
    measureMean = np.mean(shownCards, axis=0)
    measureMeanMean = measureMean.mean()
    print(subjectMean)
    print(measureMean)

    SS_time = 0
    for i in range(measureMean.__len__()):
        SS_time = SS_time + ((measureMean[i] -measureMeanMean)*(measureMean[i] -measureMeanMean))
    SS_time = SS_time*20
    print("SS_time: "+str(SS_time))


    SS_w = 0;
    for i in range(0,8):
        for j in range(0,20):
            SS_w = SS_w + ((shownCards[j,i] - measureMean[i])*(shownCards[j,i] - measureMean[i]))
    print("SS_w: "+str(SS_w))

    SS_subjects = 0
    for i in range(subjectMean.__len__()):
        SS_subjects = SS_subjects + ((subjectMean[i] -measureMeanMean)*(subjectMean[i] -measureMeanMean))
    SS_subjects = SS_subjects*8
    print("SS_subject: " + str(SS_subjects))
    SS_error = SS_w -SS_subjects
    print("SS_error: "+ str(SS_error))
    MS_time = SS_time/7
    print("MS_time: " + str(MS_time))
    MS_error = SS_error/(19*7)
    print("MS_error: " + str(MS_error))
    F_statistic = MS_time/MS_error
    print("F_statistic: " + str(F_statistic))
    p_value = 0.52
    print("Critical value - alfa: 2.0791")
    print("Alfa: 0.05")


def getBetaValues():
    print("BETA")

    allBeta = np.ones((20,8))
    for i in range(valuesPZAll.__len__()):
        arrayPom = valuesPZAll[i]
        position = 0;
        for j in range(1,arrayPom.__len__(),2):
            allBeta[i,position] = arrayPom[j]
            position = position +1

    betaMans = np.ones((10,8))
    manPos = 0
    betaWomans = np.ones((10,8))
    womanPos = 0

    for i in range(0,allBeta.__len__()):
        if i in manIndex:
            for j in range(0,allBeta[i].__len__()):
                betaMans[manPos,j]=allBeta[i,j]
            manPos = manPos+1
        else:
            for j in range(0,allBeta[i].__len__()):
                betaWomans[womanPos,j]=allBeta[i,j]
            womanPos = womanPos+1

    allMeansPZman = np.mean(betaMans, axis=1)
    meanPZMan = np.mean(allMeansPZman)

    allMeansPZwoman = np.mean(betaWomans, axis=1)
    meanPZWoman = np.mean(allMeansPZwoman)
    print(meanPZMan)
    print(meanPZWoman)

    print("ALPHA")
    allAlpha = np.ones((20, 8))
    for i in range(valuesPZAll.__len__()):
        arrayPom = valuesPZAll[i]
        position = 0;
        for j in range(0, arrayPom.__len__(), 2):
            allAlpha[i, position] = arrayPom[j]
            position = position + 1

    AlphaMans = np.ones((10, 8))
    manPos = 0
    AlphaWomans = np.ones((10, 8))
    womanPos = 0

    for i in range(0, allAlpha.__len__()):
        if i in manIndex:
            for j in range(0, allAlpha[i].__len__()):
                AlphaMans[manPos, j] = allAlpha[i, j]
            manPos = manPos + 1
        else:
            for j in range(0, allAlpha[i].__len__()):
                AlphaWomans[womanPos, j] = allAlpha[i, j]
            womanPos = womanPos + 1

    allMeansPZmanAlpha = np.mean(AlphaMans, axis=1)
    meanPZManAlpha = np.mean(allMeansPZmanAlpha)

    allMeansPZwomanAlpha = np.mean(AlphaWomans, axis=1)
    meanPZWomanAlpha = np.mean(allMeansPZwomanAlpha)
    print(meanPZManAlpha)
    print(meanPZWomanAlpha)

    # creating the dataset
    data = {'M': meanPZManAlpha, 'W': meanPZWomanAlpha}
    courses = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(courses, values,
            width=0.9)

    plt.xlabel("Freq")
    plt.title("Mans/Woman brain activity in Alpha")
    plt.show()




    data = {'M': meanPZMan, 'W': meanPZWoman}
    courses = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(courses, values,
            width=0.9)

    plt.xlabel("Freq")
    plt.title("Mans/Woman brain activity in Beta")
    plt.show()




def main():
    mne.sys_info()
    print("python main function")
    pos= 0;
    for i in range(0,20):
        path = "D:/ZCU/5.rocnik/OborovyProjekt/DP Klára Beránková/EEG data/NeupravenaEEGdata/Subjekt " + str(i+1)
        os.chdir(path)
        pos = 0;
        for file in glob.glob("*.vhdr"):
            showData(path+"/"+file, pos,i)
            pos = pos+2

    print(valuesPZAll)

    reworkExcel()
    print(shownCards)
    print(succesfull)
    print(heartBeat)
    means()
    getBetaValues()




"""
    channel_names = ['Fz', 'Cz']
    two_meg_chans = mneraw[channel_names, start_sample:stop_sample]
    y_offset = np.array([5e-11, 0])  # just enough to separate the channel traces
    x = two_meg_chans[1]
    y = two_meg_chans[0].T + y_offset
    lines = plt.plot(x, y)
    plt.legend(lines, channel_names)
    plt.
show()
"""
if __name__ == '__main__':
    main()