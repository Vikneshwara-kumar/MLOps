from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
from preprocessing import load_and_preprocess_data
from build_model import build_model_lstm as build_model
import mlflow
import mlflow.sklearn
from evaluate import evaluate 



def train(file_path):
    # Define parameters
    batch_size = 32
    epochs = 1
    validation_split = 0.1

    # Load and preprocess data
    X_train, X_test, Y_train, Y_test, scaler = load_and_preprocess_data(file_path)

    # Build the model
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Log hyperparameters
    #mlflow.log_param("C", 1.0)
    #mlflow.log_param("max_iter", 100)
    #mlflow.log_param("epochs", epochs)
    #mlflow.log_param("batch_size", batch_size)
    #mlflow.log_param("learning_rate", 0.001)
    #mlflow.log_param("loss", 0.001)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

    # Train the model
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping, reduce_lr])

    #Evaluate
    accuracy = evaluate(model, X_test, Y_test)

    # Log accuracy metric
    #mlflow.log_metric("accuracy", accuracy)
    ## Log the model as an artifact
    #mlflow.sklearn.log_model(model, "LSTM")

    # Save the model
    model.save('/model/model.keras')

    return model, accuracy