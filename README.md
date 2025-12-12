# Heartbeat Sound Classification using CNN, LSTM, and Hybrid Deep Learning Models

This project implements an end-to-end deep learning pipeline to classify heartbeat (phonocardiogram) sounds into **Normal**, **Murmur**, and **Artifact** categories. The workflow includes EDA, MFCC extraction, class balancing, and model development using **CNN**, **BiLSTM**, and a **Hybrid CNN–BiLSTM** architecture. The system helps in automated cardiac anomaly detection for healthcare and diagnostic applications.



##  Features
- Heartbeat audio EDA (waveform, spectrum, spectrogram, MFCCs)
- MFCC-based feature extraction
- Class imbalance correction using class weights
- CNN model for local feature extraction
- BiLSTM model for temporal pattern learning
- Hybrid CNN + BiLSTM model for enhanced accuracy
- Training history visualization
- Classification report & model evaluation
- Saved trained models 



##  Dataset Structure
The dataset includes:
- `normal/`
- `murmur/`
- `artifact/`
- `extrahls/`
- `extrastole/`
- `unlabel/` (test audio without labels)

Extrahls + Extrastole are grouped into Normalclass.



##  Models Implemented
###  1. Convolutional Neural Network (CNN)
- Learns local frequency patterns  
- Suitable for MFCC feature maps

###  2. Bidirectional LSTM (BiLSTM)
- Learns temporal dependencies in heart rhythms

###  3. Hybrid Model (BiLSTM + CNN)
- Combines sequential & spatial pattern learning  
- Best performance among all models  



##  Training & Evaluation
- Input features: **25 MFCCs**
- Train/Test split with validation set
- Class weights used for imbalance correction
- Metrics: `Accuracy`, `Loss`, `Classification Report`

Results include:
- Model accuracy
- Confusion matrix
- Predicted labels on test data

---

## Technologies Used
- Python  
- TensorFlow/Keras  
- Librosa  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  

---

## Output
- `heart_sounds_cnn.h5` – CNN model  
- `heart_sounds.h5` – Hybrid model  
- Accuracy metrics & training plots  
- Predicted labels for test audio  



