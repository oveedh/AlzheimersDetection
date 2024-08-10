# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.

The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al

Adapted from https://github.com/NeuroTechX/bci-workshop
"""

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

# Handy little enum to make code more readable


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """
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
EEG_MEASUREMENT_TIME = 60

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


            #print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])
            
            #print(' Alpha: ', band_powers[Band.Alpha])

            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
            # These metrics could also be used to drive brain-computer interfaces

            # Alpha Protocol:
            # Simple redout of alpha power, divided by delta waves in order to rule out noise
            #alpha_metric = smooth_band_powers[Band.Alpha] / \
            #    smooth_band_powers[Band.Delta]
            #print('Alpha Relaxation: ', alpha_metric)

            # Beta Protocol:
            # Beta waves have been used as a measure of mental activity and concentration
            # This beta over theta ratio is commonly used as neurofeedback for ADHD
            # beta_metric = smooth_band_powers[Band.Beta] / \
            #     smooth_band_powers[Band.Theta]
            # print('Beta Concentration: ', beta_metric)

            # Alpha/Theta Protocol:
            # This is another popular neurofeedback metric for stress reduction
            # Higher theta over alpha is supposedly associated with reduced anxiety
            # theta_metric = smooth_band_powers[Band.Theta] / \
            #     smooth_band_powers[Band.Alpha]
            # print('Theta Relaxation: ', theta_metric)

    except KeyboardInterrupt:
        print('Closing!')

#combined_ch_buffer.append(all_4_channel_eeg_data)
combined_ch_buffer = np.array(all_4_channel_eeg_data)

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
    #print("inside wavelt length of beta and alpha", len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    cxy, _ = coherence(signal1, signal2, nperseg=min_len)
    return cxy

def write_to_file(eeg_data, filename=EEG_DATA_FILE_PATH):
    np.savetxt(filename, eeg_data, delimiter=',')

#live_eeg_data = np.array(live_eeg_data).reshape(-1, len(INDEX_CHANNEL))
#coherence_beta_alpha_from_live_eeg_data = wavelet_coherence(band_powers[Band.Beta], band_powers[Band.Alpha])
#coherence_beta_alpha_from_live_eeg_data = 0
#np.array(band_powers).append(np.mean(coherence_beta_alpha_from_live_eeg_data))

#write_to_file()
#print(f"Data written to" + EEG_DATA_FILE_PATH)


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
                            #features.append([np.mean(beta), np.mean(alpha), np.mean(theta), np.mean(delta), np.mean(coherence_beta_alpha)])
                            features.append([np.mean(beta), np.mean(alpha), np.mean(theta), np.mean(delta)])
                            labels.append(label)
                            patients[(patient, condition, state)].append(len(features) - 1)
                            patient_labels[patient] = (condition, state, label)
    return np.array(features), np.array(labels), patients, patient_labels

    # Load your trained model (replace 'model.pkl' with your model file)
    #model = joblib.load('/Users/oveedharwadkar/BCI/muse-lsl/examples/classify.py')


##def predict(eeg_data):
##    y_pred = clf.predict(eeg_data)
    #roc_auc = roc_auc_score(y_test, y_pred)
    #accuracy = accuracy_score(y_test, y_pred)

    #print(f'ROC AUC Score: {roc_auc}')
    #print(f'Accuracy: {accuracy}')
##    return y_pred


# Extract features using Welch's method
def extract_features(raw):
    psd, freqs = welch(raw.get_data(), fs=256, nperseg=128)
    features = np.log(psd)
    return features.flatten()

# Preprocess and extract features from the live EEG data
#features = extract_features(band_powers[Band.Alpha])

# Scale features to match the training data
#scaled_features = scale(features, axis=0)

# Reshape for model prediction
#X_live = scaled_features.reshape(1, -1)

""" Train the Model """
# Load your EEG data
base_path = '/Users/oveedharwadkar/Downloads/EEG_data'
features, labels, patients, patient_labels = load_eeg_data(base_path)
#print(features, labels)
# print(features.itemsize)
# Train-test split

print("Shape of Full Training data", features.shape)

X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.2, random_state=42)
print("Shape of Train data", X_train.shape)
print("Shape of Test data", X_test.shape)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# Predict and evaluate

y_pred = clf.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f'ROC AUC Score: {roc_auc}')
print(f'Accuracy: {accuracy}')

'''
# Patient-level predictions
# Patient-level predictions
patient_predictions = {}
for (patient, condition, state), indices in patients.items():
    patient_feature_indices = indices
    patient_features = features[patient_feature_indices]
    patient_preds = clf.predict(patient_features)
    avg_prediction = np.mean(patient_preds)
    patient_predictions[(patient, condition, state)] = avg_prediction > 0.5

# Print patient-level predictions
healthy_patients = []
ad_patients = []

for (patient, condition, state), prediction in patient_predictions.items():
    if prediction:
        ad_patients.append((patient, condition, state))
    else:
        healthy_patients.append((patient, condition, state))

'''

#    print("Healthy Patients:")
#    for patient, condition, state in healthy_patients:
#        print(f'{patient}({condition})({state})')

#    print("\nPatients with Alzheimer’s Disease:")
#    for patient, condition, state in ad_patients:
#        print(f'{patient}({condition})({state})')

#prediction = clf.predict(eeg_data)
#return prediction
# Predict using the trained model

scaler = StandardScaler()
#predict AD patient
features1 = []
labels1 = []
eeg_signal_test_ad = np.loadtxt('/Users/oveedharwadkar/Downloads/TrialData/AD/AD_Patient_9_Fp1.txt')
#reshaped_eeg_test_ad = eeg_signal_test_ad.reshape(-1,1)
beta_ad, alpha_ad, theta_ad, delta_ad = preprocess_eeg(eeg_signal_test_ad)
#print("AD Beta and alpha ", beta_ad, alpha_ad)
#coherence_beta_alpha = wavelet_coherence(beta_ad, alpha_ad)
#features1.append([np.mean(beta_ad), np.mean(alpha_ad), np.mean(theta_ad), np.mean(delta_ad), np.mean(coherence_beta_alpha)])
features1.append([np.mean(beta_ad), np.mean(alpha_ad), np.mean(theta_ad), np.mean(delta_ad)])
labels1.append("AD Patient")
patients = "Paciente 9"
patient_labels = "AD EyesOpen AD Patient"
print("Shape of AD Patient 9 data", np.array(features1).shape)
prediction1 = clf.predict(np.array(features1))
print("AD Alzheimer's Prediction:", prediction1)

#predict healthy patient
features2 = []
labels2 = []
eeg_signal_test_healthy = np.loadtxt('/Users/oveedharwadkar/Downloads/TrialData/Healthy/Healthy_Patient_9_FP1.txt')
beta, alpha, theta, delta = preprocess_eeg(eeg_signal_test_healthy)
#coherence_beta_alpha = wavelet_coherence(beta, alpha)
#features2.append([np.mean(beta), np.mean(alpha), np.mean(theta), np.mean(delta), np.mean(coherence_beta_alpha)])
features2.append([np.mean(beta), np.mean(alpha), np.mean(theta), np.mean(delta)])
labels2.append("Healthy Patient")
patients = "Paciente 9"
patient_labels = "Healthy EyesOpen Healthy Patient"

#prediction2 = clf.predict(scale((np.array(features2))),axis=0)
#prediction2 = clf.predict(scaler.fit_transform(np.array(features2)))
print("Shape of Healthy Patient 9 Data", np.array(features2).shape)
prediction2 = clf.predict(np.array(features2))

print("Healthy Alzheimer's Prediction:", prediction2)

#prediction = predict(band_powers[Band.Alpha])

# Output the result


# Load the live EEG data
#live_eeg_data = np.loadtxt(EEG_DATA_FILE_PATH)

# Preprocess the live EEG data
#beta, alpha, theta, delta = preprocess_eeg()
#print("length of beta and alpha", len(band_buffer[Band.Beta]), len(band_buffer[Band.Alpha]))
#print("length of eeg_buffer beta and alpha", len(eeg_buffer[Band.Beta]), len(eeg_buffer[Band.Alpha]))
#print ("length of ch_data, eeg_buffer, Band buffer, band_power", len(ch_data), len(eeg_buffer), len(band_buffer), len(band_powers))



# Extract features from the live data
#live_features = np.array([[np.mean(beta), np.mean(alpha), np.mean(theta), np.mean(delta), np.mean(coherence_beta_alpha)]])
#live_features = np.array([[np.mean(band_powers[Band.Beta]), np.mean(band_powers[Band.Alpha]), np.mean(band_powers[Band.Theta]), np.mean(band_powers[Band.Delta]), np.mean(band_powers[Band.Theta])]])
#coherence_beta_alpha = wavelet_coherence(band_buffer[Band.Beta], band_buffer[Band.Alpha])
#live_band_buffer = np.array([[np.mean(band_buffer[Band.Beta]), np.mean(band_buffer[Band.Alpha]), np.mean(band_buffer[Band.Theta]), np.mean(band_buffer[Band.Delta]), np.mean(coherence_beta_alpha)]])
live_band_buffer = np.array([[np.mean(band_buffer[Band.Beta]), np.mean(band_buffer[Band.Alpha]), np.mean(band_buffer[Band.Theta]), np.mean(band_buffer[Band.Delta])]])
#print("Shape of Live Band buffer data", live_band_buffer.shape)
prediction_live = clf.predict(live_band_buffer)
print("band buffer prediction:", prediction_live)

#coherence_beta_alpha = wavelet_coherence(eeg_buffer[Band.Beta], eeg_buffer[Band.Alpha])
#live_eeg_buffer = np.array([[np.mean(eeg_buffer[Band.Beta]), np.mean(eeg_buffer[Band.Alpha]), np.mean(eeg_buffer[Band.Theta]), np.mean(eeg_buffer[Band.Delta]), np.mean(coherence_beta_alpha)]])
live_eeg_buffer = np.array([[np.mean(eeg_buffer[Band.Beta]), np.mean(eeg_buffer[Band.Alpha]), np.mean(eeg_buffer[Band.Theta]), np.mean(eeg_buffer[Band.Delta])]])
# Predict with the mode
#print("Shape of Live EEG buffer data", live_eeg_buffer.shape)
prediction_live = clf.predict(live_eeg_buffer)
print("eeg buffer prediction:", prediction_live)

#coherence_beta_alpha = wavelet_coherence(band_powers[Band.Beta], band_powers[Band.Alpha])
#live_eeg_buffer = np.array([[np.mean(band_powers[Band.Beta]), np.mean(band_powers[Band.Alpha]), np.mean(band_powers[Band.Theta]), np.mean(band_powers[Band.Delta]), np.mean(coherence_beta_alpha)]])
# Predict with the mode
#prediction_live = clf.predict(live_eeg_buffer)
#print("Live Alzheimer's Prediction3:", prediction_live)

#coherence_beta_alpha = wavelet_coherence(ch_data[Band.Beta], ch_data[Band.Alpha])
#coherence_beta_alpha = wavelet_coherence(ch_data, ch_data)
#ch_data_test = np.array([[np.mean(ch_data[Band.Beta]), np.mean(ch_data[Band.Alpha]), np.mean(ch_data[Band.Theta]), np.mean(ch_data[Band.Delta]), np.mean(coherence_beta_alpha)]])
ch_data_test = np.array([[np.mean(ch_data[Band.Beta]), np.mean(ch_data[Band.Alpha]), np.mean(ch_data[Band.Theta]), np.mean(ch_data[Band.Delta])]])
# Predict with the mode
#print("Shape of CH Data", ch_data_test.shape)
prediction_live = clf.predict(ch_data_test)
print("left ear:", prediction_live)

#coherence_beta_alpha = wavelet_coherence(LF_ch_data, LF_ch_data)
LF_data_test = np.array([[np.mean(LF_ch_data[Band.Beta]), np.mean(LF_ch_data[Band.Alpha]), np.mean(LF_ch_data[Band.Theta]), np.mean(LF_ch_data[Band.Delta])]])
# Predict with the mode
#print("Shape of LF Data", LF_data_test.shape)
prediction_live = clf.predict(LF_data_test)
print("left forehead:", prediction_live)

#coherence_beta_alpha = wavelet_coherence(RF_ch_data, RF_ch_data)
RF_data_test = np.array([[np.mean(RF_ch_data[Band.Beta]), np.mean(RF_ch_data[Band.Alpha]), np.mean(RF_ch_data[Band.Theta]), np.mean(RF_ch_data[Band.Delta])]])
# Predict with the mode
prediction_live = clf.predict(RF_data_test)
print("right forehead:", prediction_live)

#coherence_beta_alpha = wavelet_coherence(RE_ch_data, RE_ch_data)
RE_data_test = np.array([[np.mean(RE_ch_data[Band.Beta]), np.mean(RE_ch_data[Band.Alpha]), np.mean(RE_ch_data[Band.Theta]), np.mean(RE_ch_data[Band.Delta])]])
# Predict with the mode
prediction_live = clf.predict(RE_data_test)
print("right ear:", prediction_live)

#coherence_beta_alpha = wavelet_coherence(combined_ch_data[Band.Beta], combined_ch_data[Band.Alpha])
#combined_data_test = np.array([[np.mean(combined_ch_data[Band.Beta]), np.mean(combined_ch_data[Band.Alpha]), np.mean(combined_ch_data[Band.Theta]), np.mean(combined_ch_data[Band.Delta]), np.mean(coherence_beta_alpha)]])
combined_data_test = np.array([[np.mean(combined_ch_data[Band.Beta]), np.mean(combined_ch_data[Band.Alpha]), np.mean(combined_ch_data[Band.Theta]), np.mean(combined_ch_data[Band.Delta])]])
# Predict with the mode
#print("Shape of Combined Data", combined_data_test.shape)
prediction_live = clf.predict(combined_data_test)
print("combined Ch Data:", prediction_live)

#coherence_beta_alpha = wavelet_coherence(combined_ch_data[Band.Beta], combined_ch_data[Band.Alpha])
combined_buffer_data_test = np.array([[np.mean(combined_ch_buffer[Band.Beta]), np.mean(combined_ch_buffer[Band.Alpha]), np.mean(combined_ch_buffer[Band.Theta]), np.mean(combined_ch_buffer[Band.Delta])]])
# Predict with the mode
#print("Shape of combined buffer Data", combined_buffer_data_test.shape)
prediction_live = clf.predict(combined_buffer_data_test)
print("combined Ch Buffer", prediction_live)

# Rest of your original code for AD and Healthy patient prediction
'''
# Predict AD patient
features = []
labels = []
eeg_signal_test_ad = np.loadtxt('/Users/oveedharwadkar/Downloads/TrialData/AD/AD_Patient_9_Fp1.txt')
beta, alpha, theta, delta = preprocess_eeg(eeg_signal_test_ad)
coherence_beta_alpha = wavelet_coherence(beta, alpha)
features.append([np.mean(beta), np.mean(alpha), np.mean(theta), np.mean(delta), np.mean(coherence_beta_alpha)])
labels.append("AD Patient")
prediction1 = clf.predict(np.array(features))
print("AD Alzheimer's Prediction:", prediction1)

# Predict healthy patient
features2 = []
labels2 = []
eeg_signal_test_healthy = np.loadtxt('/Users/oveedharwadkar/Downloads/TrialData/Healthy/Healthy_Patient_9_F1.txt')
beta, alpha, theta, delta = preprocess_eeg(eeg_signal_test_healthy)
coherence_beta_alpha = wavelet_coherence(beta, alpha)
features2.append([np.mean(beta), np.mean(alpha), np.mean(theta), np.mean(delta), np.mean(coherence_beta_alpha)])
scaler = StandardScaler()
prediction2 = clf.predict(scaler.fit_transform(np.array(features2)))
print("Healthy Alzheimer's Prediction:", prediction2)
'''
