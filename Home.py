import streamlit as st

# Title of the homepage
st.title("AI Fermentation Optimizer")

# Create three columns
col1, col2, col3 = st.columns(3)

# Feature 1
with col1:
    st.image("https://via.placeholder.com/150", use_column_width=True) 
    st.subheader("Alcohol Forecasting")
    st.write("Forecast the next 3 hours alcohol percentage in the fermentation process")

# Feature 2
with col2:
    st.image("https://via.placeholder.com/150", use_column_width=True)
    st.subheader("Anomaly Detection")
    st.write("Detect abnormalities in the fermentation process to prevent contamination using fermentation images from the tank")

# Feature 3
with col3:
    st.image("https://via.placeholder.com/150", use_column_width=True)
    st.subheader("Infection Detection")
    st.write("Dectect infections in the fermentation process to prevent contamination using yeast cell images")