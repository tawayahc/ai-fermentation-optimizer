import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras import backend as K

# Custom MSE function for model loading
def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

# Initialize the session state for storing the DataFrame
if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = pd.DataFrame(columns=[
        'Timestamp', 'Branch', 'Tank', 'Brix', 'pH', 'Total cell count', 'Budding ratio', 'TSAI', 
        '%Alcohol', '%Acidity', 'Temp', 'Level', 'Viability'
    ])

# Load AI models based on the selected tank
def load_model_by_tank(tank):
    if tank == 'R411':
        return load_model('models/R411_alcohol_model.h5', custom_objects={'mse': mse})
    elif tank == 'R412':
        return load_model('models/R412_alcohol_model.h5', custom_objects={'mse': mse})
    elif tank == 'R421':
        return load_model('models/R421_alcohol_model.h5', custom_objects={'mse': mse})
    elif tank == 'R422':
        return load_model('models/R422_alcohol_model.h5', custom_objects={'mse': mse})

# Prepares data for LSTM input
def prepare_lstm_data(df, n_steps):
    X = []
    for i in range(len(df) - n_steps + 1):
        X.append(df.iloc[i:i + n_steps].values)
    return np.array(X)

# Page title
st.title("📈 Alocohol Forecasting")

# Labels for the input fields
labels = ['Brix', 'pH', 'Total cell count', 'Budding ratio', 'TSAI', 
            '%Alcohol', '%Acidity', 'Temp', 'Level', 'Viability']

# Create two columns for inputs
col1, col2 = st.columns(2)

# Create a list to store the user's inputs
inputs = []

# First 5 inputs in the first column
with col1:
    branch = st.selectbox("Branch", ['PK1', 'PK2', 'KS', 'KN', 'DC', 'MCE'])
    tank = st.selectbox("Tank", ['R411', 'R412', 'R421', 'R422'])
    for i in range(4):
        value = st.number_input(f"{labels[i]}", value=0.0)
        inputs.append(value)

# Next 5 inputs in the second column
with col2:
    for i in range(4, 10):
        value = st.number_input(f"{labels[i]}", value=0.0)
        inputs.append(value)

# Submit button to add the input values to the DataFrame
if st.button("Submit Input Values"):
    # Get the current date and time
    current_datetime = datetime.now()

    # Create a new row with the input values and the current timestamp
    new_row = pd.DataFrame({
        'Timestamp': [current_datetime],  # Add the timestamp as a separate column
        'Branch': [branch],
        'Tank': [tank],
        'Brix': [inputs[0]],
        'pH': [inputs[1]],
        'Total cell count': [inputs[2]],
        'Budding ratio': [inputs[3]],
        'TSAI': [inputs[4]],
        '%Alcohol': [inputs[5]],
        '%Acidity': [inputs[6]],
        'Temp': [inputs[7]],
        'Level': [inputs[8]],
        'Viability': [inputs[9]]
    })

    # Append the new row to the existing DataFrame in session state
    st.session_state['dataframe'] = pd.concat([st.session_state['dataframe'], new_row], ignore_index=True)
    st.success("Input values submitted successfully!")

# Button to run the model based on the submitted values
if st.button("Run Model"):
    if len(st.session_state['dataframe']) < 50:
        st.warning("Not enough data to run the model. Please submit more input values.")
    else:
        # Load the appropriate AI model based on the selected tank
        model = load_model_by_tank(tank)

        # Prepare data for model prediction
        n_steps = 50  # Assuming this is the same as in the model training
        new_data = st.session_state['dataframe']  # Use the entire DataFrame
        new_data_no_time = new_data.drop(columns=['Timestamp', 'Branch', 'Tank'])  # Drop non-numerical columns

        # Initialize and fit scaler
        scaler = MinMaxScaler()
        scaler.fit(new_data_no_time)

        # Scale the data
        new_data_scaled = scaler.transform(new_data_no_time)
        new_data_scaled_df = pd.DataFrame(new_data_scaled, columns=new_data_no_time.columns)

        # Prepare data for LSTM
        X_new = prepare_lstm_data(new_data_scaled_df, n_steps)

        # Reshape X data for LSTM input
        X_new = X_new.reshape((X_new.shape[0], X_new.shape[1], X_new.shape[2]))

        # Make predictions
        predictions_scaled = model.predict(X_new)

        # Reverse scaling of predictions
        alcohol_index = new_data_no_time.columns.get_loc('%Alcohol')
        dummy = np.zeros((len(predictions_scaled), new_data_no_time.shape[1]))
        dummy[:, alcohol_index] = predictions_scaled.flatten()
        predictions_unscaled = scaler.inverse_transform(dummy)[:, alcohol_index]

        # Display the prediction results
        st.write(f"Predicted %Alcohol values for {tank}:")
        st.write(predictions_unscaled)

# Display the updated DataFrame
st.dataframe(st.session_state['dataframe'])

# Optionally, download the DataFrame as a CSV file
csv = st.session_state['dataframe'].to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name='input_dataframe.csv',
    mime='text/csv',
)
