import os
from functools import partial
import multiprocessing.pool
import numpy as np
import re
import cv2
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import threading
import warnings
import pandas as pd
import csv
from collections import Counter
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from typing import List, Tuple
from scipy.signal import butter, resample
from inference_preprocess import preprocess_raw_video, detrend
from scipy.signal import find_peaks, stft, lfilter, butter, welch, convolve
import scipy.signal as signal

from scipy.interpolate import interp1d, PchipInterpolator
from scipy.fftpack import fft, ifft, fftfreq


try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

from scipy import interpolate, signal
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
import matplotlib.patches as mpatches
from collections import OrderedDict

from matplotlib import style
from scipy import signal
import matplotlib.pyplot as plt

style.use('ggplot')

def frequencyDomain1(timerIBI, fs):
    ibi = timerIBI * 1000
    steps = 1 / fs

    # create interpolation function based on the rr-samples.
    x = np.cumsum(ibi) / 1000.0
    f = interp1d(x, ibi, kind='cubic')

    # now we can sample from interpolation function
    xx = np.arange(1, np.max(x), steps)
    ibi_interpolated = f(xx)

    # second part
    fxx, pxx = signal.welch(x=ibi_interpolated, fs=fs)

    '''
    Segement found frequencies in the bands 
     - Very Low Frequency (VLF): 0-0.04Hz 
     - Low Frequency (LF): 0.04-0.15Hz 
     - High Frequency (HF): 0.15-0.4Hz
    '''
    cond_vlf = (fxx >= 0) & (fxx < 0.04)
    cond_lf = (fxx >= 0.04) & (fxx < 0.15)
    cond_hf = (fxx >= 0.15) & (fxx < 0.4)

    # calculate power in each band by integrating the spectral density
    vlf = np.trapz(pxx[cond_vlf], fxx[cond_vlf])
    lf = np.trapz(pxx[cond_lf], fxx[cond_lf])
    hf = np.trapz(pxx[cond_hf], fxx[cond_hf])

    # sum these up to get total power
    total_power = vlf + lf + hf

    # find which frequency has the most power in each band
    peak_vlf = fxx[cond_vlf][np.argmax(pxx[cond_vlf])]
    peak_lf = fxx[cond_lf][np.argmax(pxx[cond_lf])]
    peak_hf = fxx[cond_hf][np.argmax(pxx[cond_hf])]

    # fraction of lf and hf
    lf_nu = 100 * lf / (lf + hf)
    hf_nu = 100 * hf / (lf + hf)

    results = {}
    results['PowerVLF'] = round(vlf, 2)
    results['PowerLF'] = round(lf, 2)
    results['PowerHF'] = round(hf, 2)
    results['PowerTotal'] = round(total_power, 2)
    results['LF/HF'] = round(lf / hf, 2)
    results['PeakVLF'] = round(peak_vlf, 2)
    results['PeakLF'] = round(peak_lf, 2)
    results['PeakHF'] = round(peak_hf, 2)
    results['FractionLF'] = round(lf_nu, 2)
    results['FractionHF'] = round(hf_nu, 2)

    return results




def panTompkins(ECG, fs, plot=1):
    """
    Inputs:
    - ECG   : [list] | [numpy.ndarray] of ECG samples
    - fs    : [int] sampling frequency
    - plot  : [1|0] optional plot of R-peak detections overlayed on ECG signal
    Outputs:
    - Rpeaks : [list] of integers indicating R peak sample number locations
    """
    if type(ECG) == list or type(ECG) is np.ndarray:
        ECG = np.array(ECG)

        # Initialize
    RRAVERAGE1 = []
    RRAVERAGE2 = []
    IWF_signal_peaks = []
    IWF_noise_peaks = []
    noise_peaks = []
    ECG_bp_peaks = np.array([])
    ECG_bp_signal_peaks = []
    ECG_bp_noise_peaks = []
    final_R_locs = []
    T_wave_found = 0

    # LOW PASS FILTERING
    # Transfer function: H(z)=(1-z^-6)^2/(1-z^-1)^2
    a = np.array([1, -2, 1])
    b = np.array([1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1])

    impulse = np.repeat(0., len(b));
    impulse[0] = 1.
    impulse_response = signal.lfilter(b, a, impulse)

    # convolve ECG signal with impulse response
    ECG_lp = np.convolve(impulse_response, ECG)
    ECG_lp = ECG_lp / (max(abs(ECG_lp)))
    delay = 12  # full convolution

    # HIGH PASS FILTERING
    # Transfer function: H(z)=(-1+32z^-16+z^-32)/(1+z^-1)
    a = np.array([1, -1])
    b = np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 32, -32, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, -1])

    impulse = np.repeat(0., len(b));
    impulse[0] = 1.
    impulse_response = signal.lfilter(b, a, impulse)

    ECG_lp_hp = np.convolve(impulse_response, ECG_lp)
    ECG_lp_hp = ECG_lp_hp / (max(abs(ECG_lp_hp)))
    delay = delay + 32

    # BAND PASS FILTER
    nyq = fs / 2
    lowCut = 0.6 / nyq  # cut off frequencies are normalized from 0 to 1, where 1 is the Nyquist frequency
    highCut = 4 / nyq
    order = 2
    # b, a = signal.butter(order, [low, high], btype='band', analog=False, output='ba')


    b, a = signal.butter(order, [lowCut, highCut], btype='bandpass')
    ECG_bp = signal.lfilter(b, a, ECG_lp_hp)

    # DIFFERENTIATION
    # Transfer function: H(z)=(1/8T)(-z^-2-2z^-1+2z^1+z^2)
    T = 1 / fs
    b = np.array([-1, -2, 0, 2, 1]) * (1 / (8 * T))
    a = 1
    # Note impulse response of the filter with a = [1] is b
    ECG_deriv = np.convolve(ECG_bp, b)
    delay = delay + 4

    # SQUARING FUNCTION
    ECG_squared = ECG_deriv ** 2

    # MOVING INTEGRATION WAVEFORM
    N = int(np.ceil(0.150 * fs))
    ECG_movavg = np.convolve(ECG_squared, (1 / N) * np.ones((1, N))[0])

    # FUDICIAL MARK ON MOVING INTEGRATION WAVEFORM
    peaks = findPeaks(ECG_movavg)

    # LEARNING PHASE 1
    # 2 second initialize phase for MIW, 25% of max amplitude considered signal, 50% of mean signal considered noise
    initializeTime = 2 * fs
    SPKI = max(ECG_movavg[:initializeTime]) * 0.25
    NPKI = np.mean(ECG_movavg[:initializeTime]) * 0.5
    THRESHOLDI1 = NPKI + 0.25 * (SPKI - NPKI)
    THRESHOLDI2 = 0.5 * THRESHOLDI1

    # 2 second initialize for filtered signal, 25% of max amplitude considered signal, 50% of mean signal considered noise
    initializeTime = 2 * fs
    SPKF = max(ECG_bp[:initializeTime]) * 0.25
    NPKF = np.mean(ECG_bp[:initializeTime]) * 0.5
    THRESHOLDF1 = NPKF + 0.25 * (SPKF - NPKF)
    THRESHOLDF2 = 0.5 * THRESHOLDF1

    peaks = peaks[peaks > initializeTime]  # ignore peaks that occur during initialization window

    for c, peak in enumerate(peaks):
        # find corresponding peaks in filtered ECG using neighborhood search window +- 0.15 seconds
        searchInterval = int(np.round(0.15 * fs))
        searchIndices = np.arange(peak - searchInterval, peak + searchInterval + 1, 1)
        # neighborhood search indices cannot be negative and cannot exceed length of filtered ECG
        if searchIndices[0] >= 0 and all(searchIndices <= len(ECG_bp)):
            ECG_bp_peaks = np.append(ECG_bp_peaks, np.where(ECG_bp == max(ECG_bp[searchIndices]))[0][0])
        else:
            ECG_bp_peaks = np.append(ECG_bp_peaks, np.where(ECG_bp == max(ECG_bp[searchIndices[0]:len(ECG_bp) - 1])))
        # LEARNING PHASE 2
        if c > 0 and c < len(ECG_bp_peaks):
            if c < 8:
                RRAVERAGE1_vec = np.diff(peaks[:c + 1]) / fs
                RRAVERAGE1_mean = np.mean(RRAVERAGE1_vec)
                RRAVERAGE1.append(RRAVERAGE1_mean)

                RR_LOW_LIMIT = 0.92 * RRAVERAGE1_mean
                RR_HIGH_LIMIT = 1.16 * RRAVERAGE1_mean
                RR_MISSED_LIMIT = 1.66 * RRAVERAGE1_mean
            else:
                RRAVERAGE1_vec = np.diff(peaks[c - 8:c + 1]) / fs
                RRAVERAGE1_mean = np.mean(RRAVERAGE1_vec)
                RRAVERAGE1.append(RRAVERAGE1_mean)

                for rr in np.arange(0, len(RRAVERAGE1_vec)):
                    if RRAVERAGE1_vec[rr] > RR_LOW_LIMIT and RRAVERAGE1_vec[rr] < RR_HIGH_LIMIT:
                        RRAVERAGE2.append(RRAVERAGE1_vec[rr])
                        if len(RRAVERAGE2) > 8:
                            del RRAVERAGE2[:len(RRAVERAGE2) - 8]

                if len(RRAVERAGE2) == 8:
                    RR_LOW_LIMIT = 0.92 * np.mean(RRAVERAGE2)
                    RR_HIGH_LIMIT = 1.16 * np.mean(RRAVERAGE2)
                    RR_MISSED_LIMIT = 1.66 * np.mean(RRAVERAGE2)
            # If irregular heart beat detected in previous 9 beats, lower signal thresholds by half to increase detection sensitivity
            current_RR_movavg = RRAVERAGE1[-1]
            if current_RR_movavg < RR_LOW_LIMIT or current_RR_movavg > RR_MISSED_LIMIT:
                # MIW thresholds
                THRESHOLDI1 = 0.5 * THRESHOLDI1
                THRESHOLDI2 = 0.5 * THRESHOLDI1
                # Filtered ECG thresholds
                THRESHOLDF1 = 0.5 * THRESHOLDF1
                THRESHOLDF2 = 0.5 * THRESHOLDF1

            # Search back triggered if current RR interval is greater than RR_MISSED_LIMIT
            currentRRint = RRAVERAGE1_vec[-1]
            if currentRRint > RR_MISSED_LIMIT:
                SBinterval = int(np.round(currentRRint * fs))
                # find local maximum in the search back interval between signal and noise thresholds
                SBdata_IWF = ECG_movavg[peak - SBinterval + 1:peak + 1]

                SBdata_IWF_filtered = np.where((SBdata_IWF > THRESHOLDI1))[0]
                SBdata_max_loc = np.where(SBdata_IWF == max(SBdata_IWF[SBdata_IWF_filtered]))[0][0]

                if len(SBdata_IWF_filtered) > 0:
                    SB_IWF_loc = peak - SBinterval + 1 + SBdata_max_loc
                    IWF_signal_peaks.append(SB_IWF_loc)
                    # update signal and noise thresholds
                    SPKI = 0.25 * ECG_movavg[SB_IWF_loc] + 0.75 * SPKI
                    THRESHOLDI1 = NPKI + 0.25 * (SPKI - NPKI)
                    THRESHOLDI2 = 0.5 * THRESHOLDI1
                    # finding corresponding search back peak in ECG bandpass using 0.15 s neighborhood search window
                    if SB_IWF_loc < len(ECG_bp):
                        SBdata_ECGfilt = ECG_bp[SB_IWF_loc - round(0.15 * fs): SB_IWF_loc]
                        SBdata_ECGfilt_filtered = np.where((SBdata_ECGfilt > THRESHOLDF1))[0]
                        SBdata_max_loc2 = np.where(SBdata_ECGfilt == max(SBdata_ECGfilt[SBdata_ECGfilt_filtered]))[0][0]

                    else:
                        SBdata_ECGfilt = ECG_bp[SB_IWF_loc - round(0.15 * fs):]
                        SBdata_ECGfilt_filtered = np.where((SBdata_ECGfilt > THRESHOLDF1))[0]
                        SBdata_max_loc2 = np.where(SBdata_ECGfilt == max(SBdata_ECGfilt[SBdata_ECGfilt_filtered]))[0][0]

                    if ECG_bp[SB_IWF_loc - round(
                            0.15 * fs) + SBdata_max_loc2] > THRESHOLDF2:  # QRS complex detected in filtered ECG
                        # update signal and noise thresholds
                        SPKF = 0.25 * ECG_bp[SB_IWF_loc - round(0.15 * fs) + SBdata_max_loc2] + 0.75 * SPKF
                        THRESHOLDF1 = NPKF + 0.25 * (SPKF - NPKF)
                        THRESHOLDF2 = 0.5 * THRESHOLDF1
                        ECG_bp_signal_peaks.append(SB_IWF_loc - round(0.15 * fs) + SBdata_max_loc2)

                        # T-WAVE AND QRS DISRCIMINATION
            if ECG_movavg[peak] >= THRESHOLDI1:
                if currentRRint > 0.20 and currentRRint < 0.36 and c > 0:
                    # Slope of current waveform (possible T wave)
                    # mean width of QRS complex: 0.06 - 0.10 sec
                    maxSlope_current = max(np.diff(ECG_movavg[peak - round(fs * 0.075):peak + 1]))
                    # slope of the waveform (most likely QRS) that preceeded it
                    maxSlope_past = max(np.diff(ECG_movavg[peaks[c - 1] - round(fs * 0.075): peaks[c - 1] + 1]))
                    if maxSlope_current < 0.5 * maxSlope_past:  # T-wave found
                        T_wave_found = 1
                        # keep track of peaks marked as 'noise'
                        IWF_noise_peaks.append(peak)
                        # update Noise levels
                        NPKI = 0.125 * ECG_movavg[peak] + 0.875 * NPKI

                if not T_wave_found:  # current peak is a signal peak
                    IWF_signal_peaks.append(peak)
                    # adjust signal levels
                    SPKI = 0.125 * ECG_movavg[peak] + 0.875 * SPKI
                    # check if corresponding peak in filtered ECG is also a signal peak
                    if ECG_bp_peaks[c] > THRESHOLDF1:
                        SPKF = 0.125 * ECG_bp[c] + 0.875 * SPKF
                        ECG_bp_signal_peaks.append(ECG_bp_peaks[c])
                    else:
                        ECG_bp_noise_peaks.append(ECG_bp_peaks[c])
                        NPKF = 0.125 * ECG_bp[c] + 0.875 * NPKF

            elif ECG_movavg[peak] > THRESHOLDI1 and ECG_movavg[peak] < THRESHOLDI2:
                # update noise thresholds
                NPKI = 0.125 * ECG_movavg[peak] + 0.875 * NPKI
                NPKF = 0.125 * ECG_bp[c] + 0.875 * NPKF

            elif ECG_movavg[peak] < THRESHOLDI1:
                # update noise thresholds
                noise_peaks.append(peak)
                NPKI = 0.125 * ECG_movavg[peak] + 0.875 * NPKI
                ECG_bp_noise_peaks.append(ECG_bp_peaks[c])
                NPKF = 0.125 * ECG_bp[c] + 0.875 * NPKF
        else:
            if ECG_movavg[peak] >= THRESHOLDI1:  # first peak is a signal peak
                IWF_signal_peaks.append(peak)
                # update signal  thresholds
                SPKI = 0.125 * ECG_movavg[peak] + 0.875 * SPKI
                if ECG_bp_peaks[c] > THRESHOLDF1:
                    SPKF = 0.125 * ECG_bp[c] + 0.875 * SPKF
                    ECG_bp_signal_peaks.append(ECG_bp_peaks[c])
                else:
                    ECG_bp_noise_peaks.append(ECG_bp_peaks[c])
                    NPKF = 0.125 * ECG_bp[c] + 0.875 * NPKF

            elif ECG_movavg[peak] > THRESHOLDI2 and ECG_movavg[peak] < THRESHOLDI1:
                # update noise thresholds
                NPKI = 0.125 * ECG_movavg[peak] + 0.875 * NPKI
                NPKF = 0.125 * ECG[c] + 0.875 * NPKF

            elif ECG_movavg[peak] < THRESHOLDI2:
                # update noise thresholds
                noise_peaks.append(peak)
                NPKI = 0.125 * ECG_movavg[peak] + 0.875 * NPKI
                ECG_bp_noise_peaks.append(ECG_bp_peaks[c])
                NPKF = 0.125 * ECG_bp[c] + 0.875 * NPKF

                # reset
        T_wave_found = 0

        # update thresholds
        THRESHOLDI1 = NPKI + 0.25 * (SPKI - NPKI)
        THRESHOLDI2 = 0.5 * THRESHOLDI1

        THRESHOLDF1 = NPKF + 0.25 * (SPKF - NPKF)
        THRESHOLDF2 = 0.5 * THRESHOLDF1

    # adjust for filter delays
    ECG_R_locs = [int(i - delay) for i in ECG_bp_signal_peaks]
    ECG_R_locs = np.unique(ECG_R_locs)

    # neighborhood search in raw ECG signal for increase accuracy of R peak detection
    for i in ECG_R_locs:
        ECG = np.array(ECG)
        searchInterval = int(np.round(0.02 * fs))
        searchIndices = np.arange(i - searchInterval, i + searchInterval + 1, 1)
        searchIndices = [i.item() for i in searchIndices]  # convert to native Python int
        final_R_locs.append(np.where(ECG[searchIndices] == max(ECG[searchIndices]))[0][0] + searchIndices[0])

    # plot ECG signal with R peaks marked
    if plot == 1:
        samples = np.arange(0, len(ECG))
        plt.plot(samples, ECG, c='b')
        plt.scatter(final_R_locs, ECG[final_R_locs], c='r', s=30)
        plt.xlabel('Sample')
        plt.ylabel('ECG')
    else:
        pass

    return final_R_locs


def findPeaks(ECG_movavg):
    """finds peaks in Integration Waveform by smoothing, locating zero crossings, and moving average amplitude thresholding"""
    # smoothing
    N = 15
    ECG_movavg_smooth = np.convolve(ECG_movavg, np.ones((N,)) / N, mode='same')
    # signal derivative
    sigDeriv = np.diff(ECG_movavg_smooth)
    # find location of zero-crossings
    zeroCross = []
    for i, c in enumerate(np.arange(len(sigDeriv) - 1)):
        if sigDeriv[i] > 0 and sigDeriv[i + 1] < 0:
            zeroCross.append(c)

    return np.array(zeroCross)


def frequencyDomain(RRints, band_type=None, lf_bw=0.11, hf_bw=0.1, plot=0):
    """ Computes frequency domain features on RR interval data
    Parameters:
    ------------
    RRints : list, shape = [n_samples,]
           RR interval data
    band_type : string, optional
             If band_type = None, the traditional frequency bands are used to compute
             spectral power:
                 LF: 0.003 - 0.04 Hz
                 HF: 0.04 - 0.15 Hz
                 VLF: 0.15 - 0.4 Hz
             If band_type is set to 'adapted', the bands are adjusted according to
             the protocol laid out in:
             Long, Xi, et al. "Spectral boundary adaptation on heart rate
             variability for sleep and wake classification." International
             Journal on Artificial Intelligence Tools 23.03 (2014): 1460002.
    lf_bw : float, optional
          Low frequency bandwidth centered around LF band peak frequency
          when band_type is set to 'adapted'. Defaults to 0.11
    hf_bw : float, optional
          High frequency bandwidth centered around HF band peak frequency
          when band_type is set to 'adapted'. Defaults to 0.1
    plot : int, 1|0
          Setting plot to 1 creates a matplotlib figure showing frequency
          versus spectral power with color shading to indicate the VLF, LF,
          and HF band bounds.
    Returns:
    ---------
    freqDomainFeats : dict
                   VLF_Power, LF_Power, HF_Power, LF/HF Ratio
    """

    # Remove ectopic beats
    # RR intervals differing by more than 20% from the one proceeding it are removed
    NNs = []
    for c, rr in enumerate(RRints):
        if abs(rr - RRints[c - 1]) <= 0.20 * RRints[c - 1]:
            NNs.append(rr)

    # Resample @ 4 Hz
    fsResamp = 4
    tmStamps = np.cumsum(NNs)  # in seconds
    f = interpolate.interp1d(tmStamps, NNs, 'cubic')
    tmInterp = np.arange(tmStamps[0], tmStamps[-1], 1 / fsResamp)
    RRinterp = f(tmInterp)

    # Remove DC component
    RRseries = RRinterp - np.mean(RRinterp)

    # Pwelch w/ zero pad
    fxx, pxx = signal.welch(RRseries, fsResamp, nfft=2 ** 14, window='hann')

    vlf = (0.003, 0.04)
    lf = (0.04, 0.15)
    hf = (0.15, 0.4)

    plot_labels = ['VLF', 'LF', 'HF']

    if band_type == 'adapted':

        vlf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= vlf[0], fxx < vlf[1])]))[0][0]]
        lf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= lf[0], fxx < lf[1])]))[0][0]]
        hf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= hf[0], fxx < hf[1])]))[0][0]]

        peak_freqs = (vlf_peak, lf_peak, hf_peak)

        hf = (peak_freqs[2] - hf_bw / 2, peak_freqs[2] + hf_bw / 2)
        lf = (peak_freqs[1] - lf_bw / 2, peak_freqs[1] + lf_bw / 2)
        vlf = (0.003, lf[0])

        if lf[0] < 0:
            print('***Warning***: Adapted LF band lower bound spills into negative frequency range')
            print('Lower thresold of LF band has been set to zero')
            print('Adjust LF and HF bandwidths accordingly')
            lf = (0, lf[1])
            vlf = (0, 0)
        elif hf[0] < 0:
            print('***Warning***: Adapted HF band lower bound spills into negative frequency range')
            print('Lower thresold of HF band has been set to zero')
            print('Adjust LF and HF bandwidths accordingly')
            hf = (0, hf[1])
            lf = (0, 0)
            vlf = (0, 0)

        plot_labels = ['Adapted_VLF', 'Adapted_LF', 'Adapted_HF']


    df = fxx[1] - fxx[0]
    vlf_power = np.trapz(pxx[np.logical_and(fxx >= vlf[0], fxx < vlf[1])], dx=df)
    lf_power = np.trapz(pxx[np.logical_and(fxx >= lf[0], fxx < lf[1])], dx=df)
    hf_power = np.trapz(pxx[np.logical_and(fxx >= hf[0], fxx < hf[1])], dx=df)
    totalPower = vlf_power + lf_power + hf_power

    # Normalize and take log
    vlf_NU_log = np.log((vlf_power / (totalPower - vlf_power)) + 1)
    lf_NU_log = np.log((lf_power / (totalPower - vlf_power)) + 1)
    hf_NU_log = np.log((hf_power / (totalPower - vlf_power)) + 1)
    lfhfRation_log = np.log((lf_power / hf_power) + 1)

    freqDomainFeats = {'VLF_Power': vlf_NU_log, 'LF_Power': lf_NU_log,
                       'HF_Power': hf_NU_log, 'LF/HF': lfhfRation_log}

    if plot == 1:
        # Plot option
        freq_bands = {'vlf': vlf, 'lf': lf, 'hf': hf}
        freq_bands = OrderedDict(sorted(freq_bands.items(), key=lambda t: t[0]))
        colors = ['lightsalmon', 'lightsteelblue', 'darkseagreen']
        fig, ax = plt.subplots(1)
        ax.plot(fxx, pxx, c='grey')
        plt.xlim([0, 0.40])
        plt.xlabel(r'Frequency $(Hz)$')
        plt.ylabel(r'PSD $(s^2/Hz$)')

        for c, key in enumerate(freq_bands):
            ax.fill_between(
                fxx[min(np.where(fxx >= freq_bands[key][0])[0]): max(np.where(fxx <= freq_bands[key][1])[0])],
                pxx[min(np.where(fxx >= freq_bands[key][0])[0]): max(np.where(fxx <= freq_bands[key][1])[0])],
                0, facecolor=colors[c])

        patch1 = mpatches.Patch(color=colors[0], label=plot_labels[2])
        patch2 = mpatches.Patch(color=colors[1], label=plot_labels[1])
        patch3 = mpatches.Patch(color=colors[2], label=plot_labels[0])
        plt.legend(handles=[patch1, patch2, patch3])
        plt.show()

    return freqDomainFeats



def timedomain(rr, rr_2):
    results = {}

    hr = 60 / rr
    nn = np.gradient(rr*1000)

    results['Mean RR (ms)'] = np.mean(rr*1000)
    results['STD RR/SDNN (ms)'] = np.std(rr*1000)
    results['Mean HR (Kubios\' style) (beats/min)'] = 60 / np.mean(rr)
    results['Mean HR (beats/min)'] = np.mean(hr)
    results['STD HR (beats/min)'] = np.std(hr)
    results['Min HR (beats/min)'] = np.min(hr)
    results['Max HR (beats/min)'] = np.max(hr)
    results['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(rr*1000))))

    results['NNxx'] = np.sum(np.abs(nn) > 50) * 1
    results['pNNxx (%)'] = 100 * np.sum((np.abs(nn) > 50) * 1) / len(nn)

    # results['NNxx'] = np.sum(np.abs(rr*1000) > 50) * 1
    # results['pNNxx (%)'] = 100 * np.sum((np.abs(rr*1000) > 50) * 1) / len(rr)
    return results

def frequency_domain(rri, band_type=None, lf_bw=0.11, hf_bw=0.1, plot=0):

    # Remove ectopic beats
    # RR intervals differing by more than 20% from the one proceeding it are removed
    NNs = []
    for c, rr in enumerate(rri):
        if abs(rr - rri[c - 1]) <= 0.20 * rri[c - 1]:
            NNs.append(rr)
    # Resample @ 4 Hz
    fsResamp = 4
    # tmStamps = np.cumsum(NNs)  # in seconds
    tmStamps = np.cumsum(rri)  # in seconds

    NNs = rri

    # f = interpolate.interp1d(tmStamps, NNs, 'linear', bounds_error=False, fill_value="extrapolate")
    f = PchipInterpolator(tmStamps, NNs, extrapolate=True)
    # f = interpolate.Akima1DInterpolator(tmStamps, NNs, axis=0)
    tmInterp = np.arange(tmStamps[0], tmStamps[-1], 1 / fsResamp)
    RRinterp = f(tmInterp)

    # Remove DC component
    RRseries = RRinterp - np.mean(RRinterp)

    # Pwelch w/ zero pad
    fxx, pxx = signal.welch(RRseries, fsResamp, nfft=2 ** 14, window='hann')

    vlf = (0.003, 0.04)
    lf = (0.04, 0.15)
    hf = (0.15, 0.4)

    plot_labels = ['VLF', 'LF', 'HF']

    if band_type == 'adapted':

        vlf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= vlf[0], fxx < vlf[1])]))[0][0]]
        lf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= lf[0], fxx < lf[1])]))[0][0]]
        hf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= hf[0], fxx < hf[1])]))[0][0]]

        peak_freqs = (vlf_peak, lf_peak, hf_peak)

        hf = (peak_freqs[2] - hf_bw / 2, peak_freqs[2] + hf_bw / 2)
        lf = (peak_freqs[1] - lf_bw / 2, peak_freqs[1] + lf_bw / 2)
        vlf = (0.003, lf[0])

        if lf[0] < 0:
            print('***Warning***: Adapted LF band lower bound spills into negative frequency range')
            print('Lower thresold of LF band has been set to zero')
            print('Adjust LF and HF bandwidths accordingly')
            lf = (0, lf[1])
            vlf = (0, 0)
        elif hf[0] < 0:
            print('***Warning***: Adapted HF band lower bound spills into negative frequency range')
            print('Lower thresold of HF band has been set to zero')
            print('Adjust LF and HF bandwidths accordingly')
            hf = (0, hf[1])
            lf = (0, 0)
            vlf = (0, 0)

        plot_labels = ['Adapted_VLF', 'Adapted_LF', 'Adapted_HF']

    df = fxx[1] - fxx[0]

    # vlf_power = np.trapz(pxx[np.logical_and(fxx >= vlf[0], fxx < vlf[1])], dx=df)
    # lf_power = np.trapz(pxx[np.logical_and(fxx >= lf[0], fxx < lf[1])], dx=df)
    # hf_power = np.trapz(pxx[np.logical_and(fxx >= hf[0], fxx < hf[1])], dx=df)
    # totalPower = vlf_power + lf_power + hf_power


    cond_vlf = (fxx >= 0) & (fxx < 0.04)
    cond_lf = (fxx >= 0.04) & (fxx < 0.15)
    cond_hf = (fxx >= 0.15) & (fxx < 0.4)

    # calculate power in each band by integrating the spectral density
    # vlf_power = np.trapz(pxx[cond_vlf], fxx[cond_vlf])*1000
    # lf_power = np.trapz(pxx[cond_lf], fxx[cond_lf])*1000
    # hf_power= np.trapz(pxx[cond_hf], fxx[cond_hf])*1000
    vlf_power = np.trapz(pxx[np.logical_and(fxx >= vlf[0], fxx < vlf[1])], dx=df)*1000
    lf_power = np.trapz(pxx[np.logical_and(fxx >= lf[0], fxx < lf[1])], dx=df)*1000
    hf_power = np.trapz(pxx[np.logical_and(fxx >= hf[0], fxx < hf[1])], dx=df)*1000
    totalPower = vlf_power + lf_power + hf_power


    # Normalize and take log
    vlf_NU_log = np.log((vlf_power / (totalPower - vlf_power)) + 1)
    lf_NU_log = np.log((lf_power / (totalPower - vlf_power)) + 1)
    hf_NU_log = np.log((hf_power / (totalPower - vlf_power)) + 1)
    lfhfRation_log = np.log((lf_power / hf_power) + 1)

    vlfn = 100 * vlf_power / (lf_power + hf_power)
    lfn = 100 * lf_power / (lf_power + hf_power)
    hfn = 100 * hf_power / (lf_power + hf_power)


    freqDomainFeats = {'VLF_Power': vlf_NU_log, 'LF_Power': lf_NU_log,
                       'HF_Power': hf_NU_log, 'LF/HF': lfhfRation_log}

    # find which frequency has the most power in each band
    vlf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= vlf[0], fxx < vlf[1])]))[0][0]]
    lf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= lf[0], fxx < lf[1])]))[0][0]]
    hf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= hf[0], fxx < hf[1])]))[0][0]]

    results = {}
    results['Power VLF (ms2)'] = vlf_power
    results['Power LF (ms2)'] = lf_power
    results['Power HF (ms2)'] = hf_power
    results['Power Total (ms2)'] =  totalPower

    results['LF/HF'] = lfhfRation_log
    results['Peak VLF (Hz)'] = vlf_peak
    results['Peak LF (Hz)'] = lf_peak
    results['Peak HF (Hz)'] = hf_peak

    results['Fraction VLF (nu)'] = vlf_NU_log
    results['Fraction LF (nu)'] = lf_NU_log
    results['Fraction HF (nu)'] = hf_NU_log

    # results['Normalized VLF (nu)'] = vlfn
    results['Normalized LF (%)'] = lfn
    results['Normalized HF (%)'] = hfn

    if plot == 1:
        # Plot option
        freq_bands = {'vlf': vlf, 'lf': lf, 'hf': hf}
        freq_bands = OrderedDict(sorted(freq_bands.items(), key=lambda t: t[0]))
        colors = ['lightsalmon', 'lightsteelblue', 'darkseagreen']
        fig, ax = plt.subplots(1)
        ax.plot(fxx, pxx, c='grey')
        plt.xlim([0, 0.40])
        plt.xlabel(r'Frequency $(Hz)$')
        plt.ylabel(r'PSD $(s^2/Hz$)')

        for c, key in enumerate(freq_bands):
            ax.fill_between(
                fxx[min(np.where(fxx >= freq_bands[key][0])[0]): max(np.where(fxx <= freq_bands[key][1])[0])],
                pxx[min(np.where(fxx >= freq_bands[key][0])[0]): max(np.where(fxx <= freq_bands[key][1])[0])],
                0, facecolor=colors[c])

        patch1 = mpatches.Patch(color=colors[0], label=plot_labels[2])
        patch2 = mpatches.Patch(color=colors[1], label=plot_labels[1])
        patch3 = mpatches.Patch(color=colors[2], label=plot_labels[0])
        plt.legend(handles=[patch1, patch2, patch3])
        plt.show()

    return results, fxx, pxx

def frequency_domain1(rri, fs = 4):
    # Estimate the spectral density using Welch's method
    fxx, pxx = signal.welch(x=rri, fs=fs)
    fxx, pxx = signal.welch(x=rri, fs=fs, window='hann')

    '''
    Segement found frequencies in the bands
     - Very Low Frequency (VLF): 0-0.04Hz
     - Low Frequency (LF): 0.04-0.15Hz
     - High Frequency (HF): 0.15-0.4Hz
    '''
    cond_vlf = (fxx >= 0) & (fxx < 0.04)
    cond_lf = (fxx >= 0.04) & (fxx < 0.15)
    cond_hf = (fxx >= 0.15) & (fxx < 0.4)

    # calculate power in each band by integrating the spectral density
    vlf = np.trapz(pxx[cond_vlf], fxx[cond_vlf])
    lf = np.trapz(pxx[cond_lf], fxx[cond_lf])
    hf = np.trapz(pxx[cond_hf], fxx[cond_hf])

    # sum these up to get total power
    total_power = vlf + lf + hf

    # find which frequency has the most power in each band
    peak_vlf = fxx[cond_vlf][np.argmax(pxx[cond_vlf])]
    peak_lf = fxx[cond_lf][np.argmax(pxx[cond_lf])]
    peak_hf = fxx[cond_hf][np.argmax(pxx[cond_hf])]

    # fraction of lf and hf
    lf_nu = 100 * lf / (lf + hf)
    hf_nu = 100 * hf / (lf + hf)

    results = {}
    results['Power VLF (ms2)'] = vlf
    results['Power LF (ms2)'] = lf
    results['Power HF (ms2)'] = hf
    results['Power Total (ms2)'] = total_power

    results['LF/HF'] = (lf / hf)
    results['Peak VLF (Hz)'] = peak_vlf
    results['Peak LF (Hz)'] = peak_lf
    results['Peak HF (Hz)'] = peak_hf

    results['Fraction LF (nu)'] = lf_nu
    results['Fraction HF (nu)'] = hf_nu

    return results, fxx, pxx
def calc_rmssd(list):
    diff_nni = np.diff(list)#successive differences
    return np.sqrt(np.mean(diff_nni ** 2))


# independent function to calculate AVRR
def calc_avrr(list):
    return sum(list) / len(list)


# independent function to calculate SDRR
def calc_sdrr(list):
    return statistics.stdev(list)


# independent function to calculate SKEW
def calc_skew(list):
    return skew(list)


# independent function to calculate KURT
def calc_kurt(list):
    return kurtosis(list)


def calc_NNx(list):
    diff_nni = np.diff(list)
    return sum(np.abs(diff_nni) > 50)


def calc_pNNx(list):
    length_int = len(list)
    diff_nni = np.diff(list)
    nni_50 = sum(np.abs(diff_nni) > 50)
    return 100 * nni_50 / length_int

def remove_outliers(rr_intervals: List[float], verbose: bool = True, low_rri: int = 300,
                    high_rri: int = 2000) -> list:
    """
    Function that replace RR-interval outlier by nan.
    Parameters
    ---------
    rr_intervals : list
        raw signal extracted.
    low_rri : int
        lowest RrInterval to be considered plausible.
    high_rri : int
        highest RrInterval to be considered plausible.
    verbose : bool
        Print information about deleted outliers.
    Returns
    ---------
    rr_intervals_cleaned : list
        list of RR-intervals without outliers
    References
    ----------
    .. [1] O. Inbar, A. Oten, M. Scheinowitz, A. Rotstein, R. Dlin, R.Casaburi. Normal \
    cardiopulmonary responses during incremental exercise in 20-70-yr-old men.
    .. [2] W. C. Miller, J. P. Wallace, K. E. Eggert. Predicting max HR and the HR-VO2 relationship\
    for exercise prescription in obesity.
    .. [3] H. Tanaka, K. D. Monahan, D. R. Seals. Age-predictedmaximal heart rate revisited.
    .. [4] M. Gulati, L. J. Shaw, R. A. Thisted, H. R. Black, C. N. B.Merz, M. F. Arnsdorf. Heart \
    rate response to exercise stress testing in asymptomatic women.
    """

    # Conversion RrInterval to Heart rate ==> rri (ms) =  1000 / (bpm / 60)
    # rri 2000 => bpm 30 / rri 300 => bpm 200
    rr_intervals_cleaned = [rri if high_rri >= rri >= low_rri else np.nan for rri in rr_intervals]

    if verbose:
        outliers_list = []
        for rri in rr_intervals:
            if high_rri >= rri >= low_rri:
                pass
            else:
                outliers_list.append(rri)

        nan_count = sum(np.isnan(rr_intervals_cleaned))
        if nan_count == 0:
            print("{} outlier(s) have been deleted.".format(nan_count))
        else:
            print("{} outlier(s) have been deleted.".format(nan_count))
            print("The outlier(s) value(s) are : {}".format(outliers_list))
    return rr_intervals_cleaned


def bandpass_butter(arr, cut_low, cut_high, rate, order=2):
    nyq = 0.5 * rate
    low = cut_low / nyq
    high = cut_high / nyq
    b, a = signal.butter(order, [low, high], btype='band', analog=False, output='ba')
    out = signal.filtfilt(b, a, arr)
    return out

def Average(lst):
    return sum(lst) / len(lst)


def slidingAvg(N, stream):
    window = []
    global y
    list1 = []
    mean = np.mean(stream)
    # win = mean
    std_dev = np.std(stream)
    std = std_dev if (std_dev < 0.2 * mean) else 0.2 * mean
    # print(mean, std)
    # median = statistics.median(stream)
    # mc = most_common(stream)

    # for the first N inputs, sum and count to get the average     max(set(lst), key=lst.count)
    for i in range(N):
        window.append(stream[i])
    val = [mean if (abs(window[0] - mean) > std_dev) else window[0]]
    list1 = [mean if (abs(window[0] - mean) > std_dev) else window[0]]

    # afterwards, when a new input arrives, change the average by adding (i-j)/N, where j is the oldest number in the window
    for i in range(N, len(stream)):
        oldest = window[0]
        window = window[1:]
        window.append(stream[i])
        # window.append(window if (abs(stream[i] - stream[i] > std_dev)) else stream[i] )
        newest = window[0]
        val1 = oldest if (abs(newest - oldest) > std_dev) else newest
        val.append(val1)

    for i in range(1, len(val)):
        val[i] = list1[i - 1] if (abs(val[i] - val[i - 1]) > std) else val[i]
        list1.append(val[i])

        x = np.arange(0, len(stream))
        # print(len(x))
    y = list1
    return y


def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


class ImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 expand_dims=True,
                 data_format=None,
                 random_mult_range=0):
        if data_format is None:
            data_format = K.image_data_format()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.random_mult_range = random_mult_range
        self.expand_dims = expand_dims

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return NumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)

    def flow_from_directory(self, directory,
                            label_dir,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            frames_per_step=4,
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False):
        return DirectoryIterator(
            directory, self,
            label_dir,
            target_size=target_size, color_mode=color_mode,
            frames_per_step=frames_per_step,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links)

    # my addition
    def change_dims(self, x):
        """expands dimenstions of a batch of images"""
        img_channel_axis = self.channel_axis  # - 1
        x = np.expand_dims(x, axis=0)
        return x

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = self.channel_axis - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * \
                    np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx = np.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 1)[0]
            zy = zx.copy()

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(
                transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(
                transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(
                transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(
                transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)

        if self.random_mult_range != 0:
            if np.random.random() < 0.5:
                x = random_mutiplication(x, self.random_mult_range)
        return x

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Fits internal statistics to some sample data.
        Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            x: Numpy array, the data to fit on. Should have rank 4.
                In case of grayscale data,
                the channels axis should have value 1, and in case
                of RGB data, it should have value 3.
            augment: Whether to fit on randomly augmented samples
            rounds: If `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Raises
            ValueError: in case of invalid input `x`.
        """
        x = np.asarray(x, dtype=K.floatx())
        if x.ndim != 4:
            raise ValueError('Input to `.fit()` should have rank 4. '
                             'Got array with shape: ' + str(x.shape))
        if x.shape[self.channel_axis] not in {1, 3, 4}:
            warnings.warn(
                'Expected input to be images (as Numpy array) '
                'following the data format convention "' + self.data_format + '" '
                                                                              '(channels on axis ' + str(
                    self.channel_axis) + '), i.e. expected '
                                         'either 1, 3 or 4 channels on axis ' +
                str(self.channel_axis) + '. '
                                         'However, it was passed an array with shape ' + str(x.shape) +
                ' (' + str(x.shape[self.channel_axis]) + ' channels).')

        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if augment:
            ax = np.zeros(tuple([rounds * x.shape[0]] +
                                list(x.shape)[1:]), dtype=K.floatx())
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax

        if self.featurewise_center:
            self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.mean = np.reshape(self.mean, broadcast_shape)
            x -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
            broadcast_shape = [1, 1, 1]
            broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
            self.std = np.reshape(self.std, broadcast_shape)
            x /= (self.std + K.epsilon())

        if self.zca_whitening:
            flat_x = np.reshape(
                x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = linalg.svd(sigma)
            self.principal_components = np.dot(
                np.dot(u, np.diag(1. / np.sqrt(s + self.zca_epsilon))), u.T)


class Iterator(object):
    """Abstract base class for image data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, frames_per_step, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.frames_per_step = frames_per_step
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, frames_per_step, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, frames_per_step=4, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while True:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size * frames_per_step) % n
            if n > current_index + batch_size * frames_per_step:
                current_batch_size = batch_size * frames_per_step
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.
    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png'):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        if data_format is None:
            data_format = K.image_data_format()
        self.x = np.asarray(x, dtype=K.floatx())

        if self.x.ndim != 4:
            raise ValueError('Input data in `NumpyArrayIterator` '
                             'should have rank 4. You passed an array '
                             'with shape', self.x.shape)
        channels_axis = 3 if data_format == 'channels_last' else 1
        if self.x.shape[channels_axis] not in {1, 3, 4}:
            warnings.warn('NumpyArrayIterator is set to use the '
                          'data format convention "' + data_format + '" '
                                                                     '(channels on axis ' +
                          str(channels_axis) + '), i.e. expected '
                                               'either 1, 3 or 4 channels on axis ' +
                          str(channels_axis) + '. '
                                               'However, it was passed an array with shape ' + str(self.x.shape) +
                          ' (' + str(self.x.shape[channels_axis]) + ' channels).')
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.image_data_generator = image_data_generator
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(
            x.shape[0], batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(
            tuple([current_batch_size] + (1,) + list(self.x.shape)[1:]), dtype=K.floatx())  # Added +(1,) +
        for i, j in enumerate(index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(
                x.astype(K.floatx()))
            x = self.image_data_generator.standardize(x)
            x = self.image_data_generator.change_dims(x)  # my addition
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(
                                                                      1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y


def _count_valid_files_in_directory(directory, white_list_formats, follow_links):
    """Count files with extension in `white_list_formats` contained in a directory.
    # Arguments
        directory: absolute path to the directory containing files to be counted
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
    # Returns
        the count of files with extension in `white_list_formats` contained in
        the directory.
    """

    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    samples = 0
    for _, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                samples += 1
    return samples


def _list_valid_filenames_in_directory(directory, white_list_formats,
                                       class_indices, follow_links):
    """List paths of files in `subdir` relative from `directory` whose extensions are in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        class_indices: dictionary mapping a class name to its index.
    # Returns
        classes: a list of class indices
        filenames: the path of valid files in `directory`, relative from
            `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be ["class1/file1.jpg", "class1/file2.jpg", ...]).
    """

    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

    classes = []
    filenames = []
    subdir = os.path.basename(directory)
    # print(subdir)
    basedir = os.path.dirname(directory)
    basedir1 = os.path.dirname(basedir)

    # print(basedir)

    for root, _, files in _recursive_list(directory):
        for fname in files:
            is_valid = False
            for extension in white_list_formats:
                if fname.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                # print(class_indices[subdir])

                classes.append(class_indices[directory])
                # add filename relative to directory
                absolute_path = os.path.join(directory, fname)
                # filenames.append(os.path.relpath(absolute_path, basedir))
                filenames.append(os.path.relpath(absolute_path, basedir1))
                # print(len(filenames))

    return sorted(classes), sorted(filenames)


class DirectoryIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, directory, image_data_generator,
                 label_dir,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 frames_per_step=4,
                 batch_size=1, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.label = label_dir
        self.frames_per_step = frames_per_step
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', 'label', 'HRV', 'emo', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'ppm'}

        # first, count the number of samples and classes
        self.samples = 0
        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(_count_valid_files_in_directory,
                                   white_list_formats=white_list_formats,
                                   follow_links=follow_links)

        if not classes:
            classes = []
            vid = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    tasks = os.path.join(directory, subdir)
                    for task in sorted(os.listdir(tasks)):
                        cls = os.path.join(tasks, task)
                        vid.append(task)
                        classes.append(cls)
                        list_im = sorted(os.listdir(cls))

        self.num_class = len(vid)
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.class_emotions = dict(zip(vid, range(len(list_im))))
        # print(list(self.class_emotions.keys()))

        self.samples = sum(pool.map(function_partial,
                                    (os.path.join(directory, subdir)
                                     for subdir in classes)))


        # nombre d'image dans chaque répertoire du DB
        self.samples1 = pool.map(function_partial,
                                 (os.path.join(directory, subdir)
                                  for subdir in classes))

        print('Found %d images belonging to %d videos.' %
              (self.samples, self.num_class))

        # second, build an index of the images in the different class subfolders
        results = []

        # label = np.loadtxt('label.csv', delimiter=',')
        # for l in label:
        # self.label.append(l)

        self.filenames = []
        self.classes_emo = []

        self.classes = np.zeros((self.samples,), dtype='int32')
        i = 0

        for dirpath in classes:
            # print(dirpath)
            results.append(pool.apply_async(_list_valid_filenames_in_directory,
                                            (dirpath, white_list_formats,
                                             self.class_indices, follow_links)))

        batches_im = []
        for res in results:
            classes, filenames = res.get()
            #print(filenames)
            self.classes[i:i + len(classes)] = classes

            # build batch of image data
            filename = filenames

            batches_im.append(filenames)
            # for j in range((i // self.frames_per_step)*self.frames_per_step):
            i += len(classes)
        # ------------------------------TO DO----------------------------------------------------
        # -------------- --------------- Apply Overlapping---------------------------------------
        ## cplasstou machi hna

        len_images = self.samples1

        classes_bp = []
        batches_ppg = []
        for subdir in sorted(os.listdir(label_dir)):
            if os.path.isdir(os.path.join(label_dir, subdir)):
                tasks = os.path.join(label_dir, subdir)
                for task in sorted(os.listdir(tasks)):
                    cls = os.path.join(tasks, task)
                    classes_bp.append(cls)
        for ti in classes_bp:
            list_dir2 = os.listdir(ti)

            # -------------- Read BP signal files-------------
            BP_file = [filename for filename in list_dir2 if filename.startswith("BP_mm")]
            for bp in BP_file:
                PPG = os.path.join(ti + '/' + bp)
                # print(PPG)
                with open(PPG, 'r') as file:
                    PPG_values = [line.rstrip('\n') for line in file]
                    batches_ppg.append(PPG_values)

        preprocessed_sig = []

        # -------------------------------Downsampling of PPG signal--------------------------------
        for filename in batches_ppg:

            preprocessed_sig.append(filename)

        ppgi = []
        print(len(preprocessed_sig), len(batches_im))

        for i in range(len(preprocessed_sig) - (len(preprocessed_sig) - len(batches_im))):
            ppg = preprocessed_sig[i]
            im = len(batches_im[i])


            xx = len(ppg) - im
            # print(xx, im, len(ppg))
            if xx > 0:
                ppg = ppg[0:im]
            elif xx < 0:
                for test in range(-xx):
                    batches_im[i].pop()
            xx = len(ppg) - len(batches_im[i])
            # print("after =", xx, len(batches_im[i]), len(ppg))

            ########################## overlapping PPG ##############################
            overlapping = self.frames_per_step
            # print(len(y))
            for k in range((len(ppg) - self.frames_per_step) // overlapping):
                batches_ppgi = ppg[k * overlapping: k * overlapping + self.frames_per_step]
                # print(len(batches_ppgi[0]))

                for l in batches_ppgi:
                    ppgi.append(float(l))


        ########################## overlapping videos ##############################

        for j in batches_im:
            for k in range((len(j) // self.frames_per_step)):
                batch = j[k * self.frames_per_step:(k + 1) * self.frames_per_step]
                self.classes_emo.append(os.path.splitext(batch[0])[0][5:-5])
                for x in batch:
                    self.filenames.append(x)
        # print(len(self.classes_emo))
        print(len(self.filenames), len(ppgi))
        #print(self.classes_emo)

        j = 0

        self.batches = np.zeros((len(self.filenames),), dtype='int32')

        pool.close()
        pool.join()
        super(DirectoryIterator, self).__init__(
            len(self.filenames), batch_size, frames_per_step, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((self.batch_size,) + (self.frames_per_step,) +
                           self.image_shape, dtype=K.floatx())  # # my addition of +(1,)
        batch_y = np.zeros((self.batch_size,), dtype="object")
        batch_x1 = np.zeros((self.frames_per_step * self.batch_size,) +
                            self.image_shape, dtype=K.floatx())  # # my addition of +(1,)

        #Generate N batches of input (Video sequences) according to the batch-size
        for j in range(self.batch_size):
            for i in range(int(len(index_array))):
                fname = self.filenames[index_array[i]]
                # fname = self.filenames[i]
                # img = cv2.cvtColor(cv2.imread(os.path.join(self.directory, fname)), cv2.COLOR_RGB2BGR)
                img = cv2.imread(os.path.join(self.directory, fname))
                x = np.asarray(img, dtype=K.floatx())

                # img = load_img(os.path.join(self.directory, fname), grayscale=False, target_size=self.target_size)
                # x = img_to_array(img, data_format=self.data_format)
                x /= 255
                # x = self.image_data_generator.random_transform(x)
                # x = self.image_data_generator.standardize(x)
                # x = self.image_data_generator.change_dims(x)  # my addition
                batch_x1[i] = x
                # batch_y[j] = self.classes_emo[j]
                # print(self.classes_emo[k])
        # batch_x = batch_x1.reshape(-1, self.frames_per_step, self.image_shape)

        batch_x = batch_x1.reshape((-1,) + (self.frames_per_step,) +
                                   self.image_shape)  # # my addition of +(1,)
        # print(os.path.join(self.directory, fname))
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(
                                                                      1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'HRV':

            batch_y = np.array(batch_y)

        elif self.class_mode == 'emo':

            batch_y = np.zeros((self.batch_size, len(list(self.class_emotions.keys()))), dtype="object")
            #batch_y1 = np.zeros((self.batch_size, len(list(self.class_emotions.keys()))), dtype="object")

            for j in range(self.batch_size):
                for k in range(len(list(self.class_emotions.keys()))):
                    for i in range(self.frames_per_step):
                        fname = self.filenames[index_array[self.frames_per_step * j]]

                    classes_emo = os.path.split(os.path.normpath(fname))[0].rsplit('/',1)[-1]

                    batch_y[j,k] = classes_emo
                    batch_y1 = (np.array(list(self.class_emotions.keys())) == batch_y).astype(int)

        else:
            return batch_x
        return batch_x, batch_y

if __name__ == '__main__':

    datagen = ImageDataGenerator()

    batch_size = 1
    # train_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/rr',
    #                                           label_dir='/home/ouzar1/Documents/pythonProject/hh',
    #                                          target_size=(160, 120), class_mode='label', batch_size=1,
    #                                          frames_per_step=100, shuffle=False)

    # train_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
    #                                           label_dir='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment-bp1',
    #                                          target_size=(160, 120), class_mode='HRV', batch_size=1,
    #                                          frames_per_step=100, shuffle=False)
    #
    train_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment/Test_set',
                                              label_dir='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/PPG/Test_data',
                                             target_size=(160, 120), class_mode='emo', batch_size=1,
                                             frames_per_step=100, shuffle=False)

    for data in train_data:
        image = data[0]
        label = data[1]
        # print(label[0].shape)
