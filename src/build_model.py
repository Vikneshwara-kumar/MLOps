from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_model_lstm(input_shape, learning_rate=0.0001):
    model = Sequential([
        LSTM(128, input_shape=input_shape),
        Dense(64, activation='relu'), 
        Dense(4, activation='softmax')  # Output layer with 4 units for 4 classes and softmax activation
    ])
    
    # Define the optimizer with a custom learning rate
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def build_model_Transformer(input_shape, num_classes, num_features, learning_rate=0.0001):
    
    # Transformer block
    # For self-attention in transformers, query, key, and value are typically the same (the input)
    attention_output = MultiHeadAttention(key_dim=num_features, num_heads=3)(input_shape, input_shape)
    attention_output = Dropout(0.1)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-5)(attention_output + input_shape)  # Add & Norm

    # Pooling layer to reduce sequence dimension
    pooled_output = GlobalAveragePooling1D()(attention_output)

    # Fully connected layers
    x = Dense(64, activation='relu')(pooled_output)
    outputs = Dense(num_classes, activation='softmax')(x)  # Output layer for multi-class classification

    # Create the model
    model = Model(inputs=input_shape, outputs=outputs)

    # Compile the model
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
