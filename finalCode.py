import joblib
import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
#from sklearn.externals import joblib
from scipy.signal import welch
from sklearn.preprocessing import scale
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
#import classify
import os
import pywt
from scipy.signal import coherence
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

'''
There are 3 key steps in this file. 
1) Get live EEG data
2) Train the classification model
3) Input the live data into the classification model for prediction
'''

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

""" STEP 1: GET LIVE EEG DATA """

# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 90

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
LE_CHANNEL = [0]
LF_CHANNEL = [1]
RF_CHANNEL = [2]
RE_CHANNEL = [3]

#Data Collection Time
EEG_MEASUREMENT_TIME = 30

EEG_DATA_FILE_PATH = '/Users/oveedharwadkar/Downloads/live_eeg_data.csv'
band_powers = []

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """
    
    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())
    
    """ 2. INITIALIZE BUFFERS """
    
    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    all_4_channel_eeg_data=[]
    combined_ch_buffer = np.zeros((int(fs * BUFFER_LENGTH), 4))
    combined_ch_data=np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))
    
    """ 3. GET DATA """
    
    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    # Define end-time equal to current time plus EEG_DATA_MEASUREMENT_TIME
    endTime = time.time() + EEG_MEASUREMENT_TIME
    live_eeg_data = []
    
    #getting live data
    # do whatever you do
    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
#        while time.time() < endTime:
         while time.time() < endTime:
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:,LE_CHANNEL]
            LE_ch_data=np.array(eeg_data)[:,LE_CHANNEL]
            LF_ch_data=np.array(eeg_data)[:,LF_CHANNEL]
            RF_ch_data=np.array(eeg_data)[:,RF_CHANNEL]
            RE_ch_data=np.array(eeg_data)[:,RE_CHANNEL]
            combined_ch_data=np.concatenate((combined_ch_data,LE_ch_data,LF_ch_data,RF_ch_data,RE_ch_data),0)
            all_4_channel_eeg_data.append(eeg_data)

            #combined_ch_buffer, filter_state = utils.update_buffer(
            #    combined_ch_buffer,combined_ch_data,notch=True,
            #    filter_state=filter_state)

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = utils.compute_band_powers(data_epoch, fs)
            band_buffer, _ = utils.update_buffer(band_buffer,
                                                 np.asarray([band_powers]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(band_buffer, axis=0)

            print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
                   ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])
            
    except KeyboardInterrupt:
        print('Closing!')

#combined_ch_buffer.append(all_4_channel_eeg_data)
combined_ch_buffer = np.array(all_4_channel_eeg_data)

""" STEP 2: TRAIN CLASSIFICIATION MODEL """

# Example function to preprocess and extract features from EEG data
def preprocess_eeg(eeg_signal):
    coeffs = pywt.wavedec(eeg_signal, 'db4', level=4,axis=0)
    beta = coeffs[1]
    alpha = coeffs[2]
    theta = coeffs[3]
    delta = coeffs[4]
    return beta, alpha, theta, delta

# Example function to calculate wavelet coherence
def wavelet_coherence(signal1, signal2):
    # Ensure signals are the same length
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    cxy, _ = coherence(signal1, signal2, nperseg=min_len)
    return cxy

# Function to load EEG data from folders
def load_eeg_data(base_path):
    features = []
    labels = []
    patients = defaultdict(list)
    patient_labels = {}
    for label, condition in enumerate(['Healthy', 'AD']):
        condition_path = os.path.join(base_path, condition)
        for state in ['Eyes_open', 'Eyes_closed']:
            state_path = os.path.join(condition_path, state)
            for patient in os.listdir(state_path):
                patient_path = os.path.join(state_path, patient)
                if os.path.isdir(patient_path):
                    for filename in os.listdir(patient_path):
                        if filename.startswith(('Fp1', 'Fp2', 'T6', 'T5', 'Fz')):
                            file_path = os.path.join(patient_path, filename)
                            eeg_signal = np.loadtxt(file_path)
                            beta, alpha, theta, delta = preprocess_eeg(eeg_signal)
                            coherence_beta_alpha = wavelet_coherence(beta, alpha)
                            features.append([np.mean(beta), np.mean(alpha), np.mean(theta), np.mean(delta)])
                            labels.append(label)
                            patients[(patient, condition, state)].append(len(features) - 1)
                            patient_labels[patient] = (condition, state, label)
    return np.array(features), np.array(labels), patients, patient_labels

# Extract features using Welch's method
def extract_features(raw):
    psd, freqs = welch(raw.get_data(), fs=256, nperseg=128)
    features = np.log(psd)
    return features.flatten()

""" Train the Model """
# Load your EEG data
base_path = '/Users/oveedharwadkar/Downloads/EEG_data'
features, labels, patients, patient_labels = load_eeg_data(base_path)
X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Accuracy
print(f'Accuracy: {accuracy}') # 86%

""" STEP 3: FINAL PREDICTION """
# combined ch buffer
combined_buffer_data_test = np.array([[np.mean(combined_ch_buffer[Band.Beta]), np.mean(combined_ch_buffer[Band.Alpha]), np.mean(combined_ch_buffer[Band.Theta]), np.mean(combined_ch_buffer[Band.Delta])]])
prediction_live = clf.predict(combined_buffer_data_test)
if prediction_live == '[1]':
    print("Patient has Alzheimers? True")
else:
    print("Patient has Alzheimers? False")

#print("Patient Has Alzheimers?", prediction_live)
