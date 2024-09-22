import streamlit as st
import pandas as pd
from datetime import datetime

# Initialize the session state for storing the DataFrame
if 'dataframe' not in st.session_state:
    st.session_state['dataframe'] = pd.DataFrame(columns=[
        'Timestamp', 'Branch', 'Tank', 'Brix', 'pH', 'Total cell count', 'Budding ratio', 'TSAI', 
        '%Alcohol', '%Acidity', 'Temp', 'Level', 'Viability'
    ])

st.title("🚨 Infection Detection")

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

# Submit button to process inputs
if st.button("Submit"):
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