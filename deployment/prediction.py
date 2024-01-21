# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load
with open('pipeline_best.pkl', 'rb') as file_1:
    pipeline_best = pickle.load(file_1)

# Define the features for the options
features = ['hotel', 'lead_time', 'arrival_date_year',
        'arrival_date_month', 'arrival_date_week_number',
        'arrival_date_day_of_month', 'stays_in_weekend_nights',
        'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
        'country', 'market_segment', 'distribution_channel',
        'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'reserved_room_type',
        'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
        'company', 'days_in_waiting_list', 'customer_type', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests',
        'reservation_status', 'reservation_status_date']

# Define the categorical features for the options
categorical_features = ['hotel', 'arrival_date_month',
        'arrival_date_day_of_month', 'meal',
        'country', 'market_segment', 'distribution_channel',
        'is_repeated_guest', 'reserved_room_type',
        'assigned_room_type', 'deposit_type', 'customer_type',
        'reservation_status']

# Define the options for the categorical features
options = {
    'hotel': ['Resort Hotel', 'City Hotel'],
    'arrival_date_month': ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    'arrival_date_day_of_month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    'meal': ['BB', 'FB', 'HB', 'SC'],
    'country': ['PRT', 'GBR', 'USA', 'ESP', 'IRL', 'FRA', '0    PRT', 'ROU', 'NOR', 'OMN', 'ARG'
    'POL', 'DEU', 'BEL', 'CHE', 'CN', 'GRC', 'ITA', 'NLD', 'DNK', 'RUS', 'SWE', 'AUS'
    'EST', 'CZE', 'BRA', 'FIN', 'MOZ', 'BWA', 'LUX', 'SVN', 'ALB', 'IND', 'CHN', 'MEX'
    'MAR', 'UKR', 'SMR', 'LVA', 'PRI', 'SRB', 'CHL', 'AUT', 'BLR', 'LTU', 'TUR', 'ZAF'
    'AGO', 'ISR', 'CYM', 'ZMB', 'CPV', 'ZWE', 'DZA', 'KOR', 'CRI', 'HUN', 'ARE', 'TUN'
    'JAM', 'HRV', 'HKG', 'IRN', 'GEO', 'AND', 'GIB', 'URY', 'JEY', 'CAF', 'CYP', 'COL'
    'GGY', 'KWT', 'NGA', 'MDV', 'VEN', 'SVK', 'FJI', 'KAZ', 'PAK', 'IDN', 'LBN', 'PHL'
    'SEN', 'SYC', 'AZE', 'BHR', 'NZL', 'THA', 'DOM', 'MKD', 'MYS', 'ARM', 'JPN', 'LKA'
    'CUB', 'CMR', 'BIH', 'MUS', 'COM', 'SUR', 'UGA', 'BGR', 'CIV', 'JOR', 'SYR', 'SGP'
    'BDI', 'SAU', 'VNM', 'PLW', 'QAT', 'EGY', 'PER', 'MLT', 'MWI', 'ECU', 'MDG', 'ISL'
    'UZB', 'NPL', 'BHS', 'MAC', 'TGO', 'TWN', 'DJI', 'STP', 'KNA', 'ETH', 'IRQ', 'HND'
    'RWA', 'KHM', 'MCO', 'BGD', 'IMN', 'TJK', 'NIC', 'BEN', 'VGB', 'TZA', 'GAB', 'GHA'
    'TMP', 'GLP', 'KEN', 'LIE', 'GNB', 'MNE', 'UMI', 'MYT', 'FRO', 'MMR', 'PAN', 'BFA'
    'LBY', 'MLI', 'NAM', 'BOL', 'PRY', 'BRB', 'ABW', 'AIA', 'SLV', 'DMA', 'PYF', 'GUY'
    'LCA', 'ATA', 'GTM', 'ASM'],
    'market_segment': ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups', 'Aviation'],
    'distribution_channel': ['Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS'],
    'is_repeated_guest': [0, 1],
    'reserved_room_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
    'assigned_room_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
    'deposit_type': ['No Deposit', 'Refundable', 'Non Refund'],
    'customer_type': ['Transient', 'Contract', 'Transient-Party', 'Group'],
    'reservation_status': ['Check-Out', 'Canceled', 'No-Show', 'C'],
    'reservation_status_date': ['2017-12-31']
}

def run():
    # Create a sidebar
    st.sidebar.title("Prediction Options")
    st.sidebar.subheader("Enter the values for the features")

    # Create inputs for the features
    inputs = {}
    for feature in features:
        if feature in categorical_features:
            inputs[feature] = st.sidebar.selectbox(feature, options[feature])
        else:
            inputs[feature] = st.sidebar.number_input(feature, min_value=0)

    # Create a button for prediction
    predict = st.sidebar.button("Predict")

    # Create a main title
    st.title("Click the `Predict` button to start")

    # Display the prediction
    if predict:
        # Convert the inputs into a dataframe
        input_df = pd.DataFrame([inputs])

        # Make the prediction
        prediction = pipeline_best.predict(input_df)[0]

        # Display the result
        if prediction == 0:
            st.success("# The client is not likely to cancel the reservation.")
        else:
            st.error("# The client is likely to cancel the reservation.")
            
if __name__ == '__main__':
    run()
