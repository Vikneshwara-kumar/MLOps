import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
from preprocessing import load_and_preprocess_data
from model import build_model_lstm as build_model
import mlflow
import mlflow.sklearn
import joblib


# Define parameters
batch_size = 32
epochs = 50
validation_split = 0.1

# File paths and columns
file_path = 'Dataset/Dataset.csv'
feature_columns = ['RMS','MAV','SSC','WL','MNF','MDF','IMDF','IMPF','PSD','MNP','ZC','stft_feature_1','stft_feature_2','stft_feature_3','stft_feature_4','stft_feature_5','stft_feature_6']
label_column = ['Label']

# Set up MLflow and track the experiment
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")  # Change if MLflow is running elsewhere
mlflow.set_experiment("Test Experiment")

with mlflow.start_run():
    # Load and preprocess data
    X_train, X_test, Y_train, Y_test, scaler = load_and_preprocess_data(file_path, feature_columns, label_column)

    # Build the model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Log hyperparameters
    mlflow.log_param("C", 1.0)
    mlflow.log_param("max_iter", 100)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("loss", 0.001)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    # Train the model
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping, reduce_lr])

    # Evaluate model
    predictions = model.predict(X_test)
    predictions_class = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(np.argmax(Y_test, axis=1), predictions_class)*100 
    print(f"Model Accuracy: : {accuracy}")
    accuracy = accuracy_score(np.argmax(Y_test, axis=1), predictions_class)
    print(f"Model Accuracy: : {accuracy}")
    print(type(accuracy))

    # Log accuracy metric
    mlflow.log_metric("accuracy", accuracy)
    # Log the model as an artifact
    mlflow.sklearn.log_model(model, "model")

