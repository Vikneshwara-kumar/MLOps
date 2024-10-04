import mlflow
from model import train

# File paths and columns
file_path = 'Dataset.csv'

if __name__ == "__main__":

    # Start an MLFlow experiment
    #mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")  # Change if MLflow is running elsewhere
    #mlflow.set_experiment("ml-ci-cd-github-experiment")
    
    #with mlflow.start_run():
        # Train and evaluate the model
    model, accuracy = train(file_path)
    print(f"Model trained with accuracy: {accuracy}")
