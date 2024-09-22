import streamlit as st

st.set_page_config(page_title="Fermentation Optimizer")


st.title("Welcome to the Home Page")
st.write("This is the home page of the Streamlit application.")
st.markdown("<hr/>")
# Create two columns for the cards
col1, col2 = st.columns(2)

# Card for Feature 1
with col1:
    st.subheader("Feature 1")
    st.write("Description of Feature 1")
    if st.button("Go to Feature 1", key='feature1'):
        st.session_state['page'] = 'Feature 1'

# Card for Feature 2
with col2:
    st.subheader("Feature 2")
    st.write("Description of Feature 2")
    if st.button("Go to Feature 2", key='feature2'):
        st.session_state['page'] = 'Feature 2'