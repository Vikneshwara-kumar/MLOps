# Use official Python image as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary dependencies
RUN pip install --upgrade pip
RUN pip install -r src/requirements.txt

# Define the command to run the training script
CMD ["python", "src/train.py"]
