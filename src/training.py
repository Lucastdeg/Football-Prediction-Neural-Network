import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the processed data
data_path = "data/features_data.csv" 
df = pd.read_csv(data_path)

df = df.drop(columns=["home_team", "away_team"], errors='ignore')  # Ignore if the columns do not exist


# Prepare features and target
X = df.drop(columns=["result"])  # All columns except the result
y = df["result"]  # Target column

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(128, activation='sigmoid', input_shape=(X_train.shape[1],)))  # Input layer with 128 neurons
model.add(Dense(64, activation='relu'))  # Hidden layer with 64 neurons
model.add(Dense(32, activation='relu'))  # Hidden layer with 32 neurons
model.add(Dense(16, activation='sigmoid'))  # Hidden layer with 16 neurons
model.add(Dense(8, activation='sigmoid'))  # Additional hidden layer with 8 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification


# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=80, batch_size=40, validation_split=0.2)

# Save the model
model.save("football_match_predictor3.h5")

print("Model training complete! Saved to model/football_match_predictor.h5")

