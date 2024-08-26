import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path, feature_columns, label_column, timesteps=30, num_classes=4):
    df = pd.read_csv(file_path)

    # Split the DataFrame into features (X) and target (y)
    X = df[feature_columns]
    Y = df[label_column]

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    data_segments = {f'X{j}': [] for j in range(X_scaled.shape[1])}  # Dictionary to store segments for each feature
    y = []  # List to store labels

    # Segment the data from df
    for i in range(0, X.shape[0] - timesteps):
        for j in range(X_scaled.shape[1]):  # Loop for column indexing
            data_segments[f'X{j}'].append(X.iloc[i:i + timesteps, j].values)

    # Segment the labels from labels_df
    for i in range(0, Y.shape[0] - timesteps):
        y.append(Y.iloc[i + timesteps, 0])  # Assuming the label to predict is the last one in the window

    y_encoded = to_categorical(y, num_classes=num_classes)

    # Convert lists in data_segments to NumPy arrays for easier manipulation
    X_segments = np.array([np.array(data_segments[f'X{j}']) for j in range(X_scaled.shape[1])])
    X_segments = np.swapaxes(X_segments, 0, 1)  # Swap axes to get the shape (samples, timesteps, features)
    X_segments = np.swapaxes(X_segments, 1, 2)  # Final shape: (samples, timesteps, features)

    # Convert labels list to a NumPy array
    Y = np.array(y_encoded)

    # Splitting the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_segments, Y, test_size=0.2, random_state=4)

    return X_train, X_test, Y_train, Y_test, scaler
