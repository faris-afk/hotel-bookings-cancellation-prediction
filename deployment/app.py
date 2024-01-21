import pandas as pd
import streamlit as st
import eda
import prediction

# Set the page title and favicon
st.set_page_config(page_title="Hotel Cancelation Prediction", 
                page_icon="🏨",
                layout='wide',
                initial_sidebar_state='expanded'
)

# Create a sidebar with a title and a selection box
st.sidebar.title("Choose a page:")
page = st.sidebar.selectbox("", ('Landing Page', 'Data Exploration', 'Data Prediction'))

# Display different content depending on the selected page
if page == 'Data Exploration':
    eda.run()
elif page == 'Data Prediction':
    prediction.run()
else:
    # Add a header and a subheader with some text
    st.title("Will a customer cancel their booked rooms?")
    st.subheader("Find out with this web app that uses machine learning to predict your booking cancelation risk.")

    # Add an image about the case
    st.image("https://images.unsplash.com/photo-1625244724120-1fd1d34d00f6?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8aG90ZWxzfGVufDB8fDB8fHww", width=300)
    with st.expander("Backgroud dataset"):
        st.caption("""
                The data set used contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things.
                """)
    with st.expander("Problem statements"):
        st.caption("The goal is to predict whether a customer will cancel their room reservation based on the given features.")
    with st.expander("Sponsor"):
        st.caption("""
                This web app is sponsored by **XYZ Tourism Company**, a leading tourism company that focuses on hotel bussiness. 

                XYZ Bank is committed to providing the best customer experience and ensuring the best focused services during peak seasons.
                """)
        st.image("data:image/png;base64,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")