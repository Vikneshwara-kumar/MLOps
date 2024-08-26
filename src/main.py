import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

from preprocessing import load_and_preprocess_data
from model import build_model_lstm as build_model



# Define parameters
batch_size = 32
epochs = 30
validation_split = 0.1

# File paths and columns
file_path = 'Dataset/Dataset.csv'
feature_columns = ['RMS','MAV','SSC','WL','MNF','MDF','IMDF','IMPF','PSD','MNP','ZC','stft_feature_1','stft_feature_2','stft_feature_3','stft_feature_4','stft_feature_5','stft_feature_6']
label_column = ['Label']

# Load and preprocess data
X_train, X_test, Y_train, Y_test, scaler = load_and_preprocess_data(file_path, feature_columns, label_column)

# Build the model
model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Train the model
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping, reduce_lr])

# Save the model
model.save('Model/model.keras')

# Save the scaler
with open('Model/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Scaler has been saved as 'Model/scaler.pkl'")
