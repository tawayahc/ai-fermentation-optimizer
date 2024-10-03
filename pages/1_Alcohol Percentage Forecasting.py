import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from keras import backend as K
import joblib
import matplotlib.pyplot as plt

# Custom MSE function for model loading
def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

# Initialize the session state for storing the DataFrame
if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = pd.DataFrame(columns=[
        'Timestamp', 'Branch', 'Tank', 'Brix', 'pH', 'Total cell count', 'Budding ratio', 'TSAI', 
        '%Alcohol', '%Acidity', 'Temp', 'Level', 'Viability'
    ])

# Load AI models and scalers based on the selected tank
def load_model_and_scalers(tank):
    if tank == 'R411':
        model = load_model('models/trained_lstm_model.h5', custom_objects={'mse': mse})
        scaler_X = joblib.load('models/scaler_X.pkl')
        scaler_y = joblib.load('models/scaler_y.pkl')
    elif tank == 'R412':
        model = load_model('models/R412_alcohol.h5', custom_objects={'mse': mse})
        scaler_X = joblib.load('models/R412_scaler_X.pkl')
        scaler_y = joblib.load('models/R412_scaler_y.pkl')
    elif tank == 'R421':
        model = load_model('models/R421_alcohol.h5', custom_objects={'mse': mse})
        scaler_X = joblib.load('models/R421_scaler_X.pkl')
        scaler_y = joblib.load('models/R421_scaler_y.pkl')
    elif tank == 'R422':
        model = load_model('models/R422_alcohol.h5', custom_objects={'mse': mse})
        scaler_X = joblib.load('models/R422_scaler_X.pkl')
        scaler_y = joblib.load('models/R422_scaler_y.pkl')
    return model, scaler_X, scaler_y

# Prepares data for LSTM input
def prepare_lstm_data(df, n_steps):
    X = []
    for i in range(len(df) - n_steps + 1):
        X.append(df.iloc[i:i + n_steps].values)
    return np.array(X)

import altair as alt

# Function to plot %Alcohol and predicted %Alcohol using Altair
def plot_alcohol_percentage(predicted_alcohol=None):
    df_plot = st.session_state['dataframe'][['Timestamp', '%Alcohol']].copy()

    # Add an "index" column for Altair's X-axis if needed
    df_plot['index'] = range(1, len(df_plot) + 1)

    # Create a DataFrame for the predicted value
    if predicted_alcohol is not None:
        predicted_df = pd.DataFrame({
            'index': [len(df_plot) + 1],
            'Timestamp': [pd.Timestamp.now()],  # Fake timestamp for the prediction point
            '%Alcohol': [predicted_alcohol],
            'Type': ['Predicted']  # To distinguish the predicted value
        })
        df_plot['Type'] = 'Historical'
        df_plot = pd.concat([df_plot, predicted_df], ignore_index=True)

    # Build the Altair chart
    chart = alt.Chart(df_plot).mark_line().encode(
        x='index:Q',
        y='%Alcohol:Q',
        color='Type:N',  # This ensures different colors for Historical and Predicted
        tooltip=['Timestamp:T', '%Alcohol:Q', 'Type:N']
    ).properties(
        width=700,
        height=400,
        # title="%Alcohol Over Time"
    )

    # Add a point marker for the predicted value
    if predicted_alcohol is not None:
        point = alt.Chart(predicted_df).mark_point(color='red', size=100).encode(
            x='index:Q',
            y='%Alcohol:Q',
            tooltip=['%Alcohol:Q']
        )
        chart = chart + point

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)


# Page title
st.title("📈 Alcohol Forecasting")

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

# Load initial data from Excel
# if st.button("Load Initial Data from Excel"):
file_path = 'data/cleaned_DC_R411.csv'
df_excel = pd.read_csv(file_path)
df_excel = df_excel.iloc[:70]  # Limit the number of rows for demonstration
df_excel.columns = ['Brix', 'pH', 'Total cell count', 'Budding ratio', 'TSAI',
                    '%Alcohol', '%Acidity', 'Temp', 'Level', 'Viability']
df_excel['Timestamp'] = pd.to_datetime('now')  # Add a timestamp
df_excel['Branch'] = 'PK1'  # Placeholder branch
df_excel['Tank'] = 'R411'  # Placeholder tank
st.session_state['dataframe'] = pd.concat([st.session_state['dataframe'], df_excel], ignore_index=True)
# st.success("Data loaded from Excel successfully!")

# Submit button to add the input values to the DataFrame
if st.button("Submit Values"):
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
if st.button("Forecast Alcohol Percentage"):
    if len(st.session_state['dataframe']) < 50:
        st.warning("Not enough data to run the model. Please submit more input values.")
    else:
        # Load the appropriate AI model and scalers based on the selected tank
        model, scaler_X, scaler_y = load_model_and_scalers(tank)

        # Prepare the input data
        new_data = st.session_state['dataframe']
        new_data_no_time = new_data.drop(columns=['Timestamp', 'Branch', 'Tank'])  # Drop non-numerical columns

        # Define the feature columns (same as used during training)
        features = ['Brix', 'pH', 'Total cell count', 'TSAI', 'Budding ratio',
                    '%Acidity', 'Temp', 'Level', 'Viability']

        # Extract the feature values and scale them
        X_input = new_data_no_time[features].values
        X_input_scaled = scaler_X.transform(X_input)

        # Reshape the input to match LSTM expected format (samples, timesteps, features)
        X_input_seq = X_input_scaled.reshape(1, X_input_scaled.shape[0], X_input_scaled.shape[1])

        # Make the prediction
        predicted_scaled = model.predict(X_input_seq)

        # Inverse transform the prediction to get the actual %Alcohol value
        predicted_alcohol = scaler_y.inverse_transform(predicted_scaled)[0][0]

        # Display the prediction results
        st.write(f"Forecasted next 3 hours Alcohol Percentage values for {tank}: {predicted_alcohol:.2f}")

        # Plot the alcohol percentage along with the prediction
        plot_alcohol_percentage(predicted_alcohol)



# Display the updated DataFrame
st.dataframe(st.session_state['dataframe'])

# Optionally, download the DataFrame as a CSV file
# csv = st.session_state['dataframe'].to_csv(index=False).encode('utf-8')
# st.download_button(
#     label="Download as CSV",
#     data=csv,
#     file_name='input_dataframe.csv',
#     mime='text/csv',
# )
