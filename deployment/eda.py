import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import seaborn as sns
import scipy.stats as stat

def run():
    # create title
    st.title('Hotel Booking Demand Dataset Exploratory Analysis')

    # add image
    st.image('https://www.fodors.com/wp-content/uploads/2019/08/05_EuropeanHotelsfromMovies__HotelLeBristol_5.-Le_Bristol_Paris_nouveau_jardin_9274.jpg')

    # add description
    st.write('This page was made to visualize my exploaration on the Hotel Booking Demand Dataset')

    # create markdown line
    st.markdown('---')

    # create dataframe
    df = pd.read_csv('hotel_bookings_clean.csv')
    st.dataframe(df.head())

    st.write('Data exploration was done by focusing on several columns')
    st.write('# 1. `country`🌏')
    st.write('Country column in the dataset explains the country of origin of each customer who made the booking.')
    st.write('## 1.1. What are the most common countries of the bookings for each hotel type?')

    # Group the data by hotel type and count the frequency of each country
    country_count = df.groupby("hotel")["country"].value_counts()

    # Sort the results in descending order
    country_count = country_count.sort_values(ascending=False)

    # Select the top countries for each hotel type
    top_countries = country_count.groupby("hotel").head()

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    top_countries.unstack().plot(kind="bar", ax=ax)

    # Add labels and title
    plt.xlabel("Hotel type")
    plt.ylabel("Number of guests")
    plt.title("Most common countries of the reservations for each hotel type")

    # Show the plot
    st.pyplot(fig)

    st.write('For the city hotel, the top countries are Portugal (PRT), France (FRA), United Kingdom (GBR), Germany (DEU), and Spain (ESP).')
    st.write('For the resort hotel, the top countries are Portugal (PRT), United Kingdom (GBR), Spain (ESP), Ireland (IRL) and France (FRA).')
    st.write('XYZ should target more guests from these top countries. They also should try to explore market potential in countries nearby those top countries, as the citizens may have similar preferences or needs. They should also increase their customer loyalty by customizing their services based on those countries cultures, for example serving sourthern europian dishes, or offering promos on those countries national holidays.')


    st.write('# 2. `arrival_date` 📅')
    st.write('the column `arrival_date_year` and `arrival_date_month` in the dataset explains the time the customers arrive.')
    st.write('## 2.1. What are the peak and low seasons for each hotel type based on the arrival date?')

    # Define the custom order of the months
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    # Convert the arrival_date_month column into a categorical variable with the custom order
    df["arrival_date_month"] = pd.Categorical(df["arrival_date_month"], categories=month_order, ordered=True)

    # Group the data by hotel type and arrival date month and count the number of bookings
    month_count = df.groupby(["hotel", "arrival_date_month"]).size()

    # Plot the line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    month_count.unstack(level=0).plot(kind="line", ax=ax)

    # Add labels and title
    plt.xlabel("Month")
    plt.ylabel("Number of bookings")
    plt.title("Seasonal patterns of demand for each hotel type")

    # Show the plot
    st.pyplot(fig)

    st.write('Both hotel has similar seasonal pattern where the peak arrival of customers are in August, while the lowest arrival of custumers are in January.')
    st.write('To adress this, the company should prepare for their services in summer season, especially in August. This recommendation also supported by the fact that summer is holiday season in most europian countries, especially Portugal.')
    st.write('To prepare apropriately, the company should also acknowledge the fact that their hotels are not popular in winter, the lowes in January, and will gradually rises again at the end of winter (February).')

    st.write('## 2.2. How do the seasons influence the cancellation rate?')

    # Create a season column using a lambda function
    df["season"] = df["arrival_date_month"].apply(lambda x: "Winter" if x in ["January", "February", "December"]
                                                                else "Spring" if x in ["March", "April", "May"]
                                                                else "Summer" if x in ["June", "July", "August"]
                                                                else "Fall")

    # Define the custom order of the seasons
    season_order = ["Spring", "Summer", "Fall", "Winter"]

    # Convert the arrival_date_season column into a categorical variable with the custom order
    df["season"] = pd.Categorical(df["season"], categories=season_order, ordered=True)

    # Group the data by hotel type and arrival date season and count the number of bookings
    season_count = df.groupby(["hotel", "season"]).size()

    # Plot the line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    season_count.unstack(level=0).plot(kind="line", ax=ax)

    # Add labels and title
    plt.xlabel("Season")
    plt.ylabel("Number of bookings")
    plt.title("Seasonal patterns of demand for each hotel type")

    # Show the plot
    st.pyplot(fig)

    st.write('In both hotel, the cancellation is highest in Summer, this makes sense as the reservations are also peaked at Summer.')
    st.write('To address this, the company may have to implement some policies or incentives to discourage cancellations, especially in the Summer seasons, when the demand is high. For example, the company can charge a cancellation fee.')

    st.write('# 3. Average Daily Rate (`adr`) 💰')
    st.write("The `adr` is the amount of money that a guest pays per night at a hotel.")
    st.write("## 3.1. Is there a significant difference in the average daily rate (`adr`) between the city hotel and the resort hotel?")
    st.write("### Hypothesis Test Results")

    # Filter the df by hotel type and cancellation status
    city = df[(df["hotel"] == "City Hotel") & (df["is_canceled"] == 0)]["adr"]
    resort = df[(df["hotel"] == "Resort Hotel") & (df["is_canceled"] == 0)]["adr"]

    # Perform the t-test
    t_stat, p_value = stat.ttest_ind(city, resort, equal_var=True)

    # Calculate the mean value of each hotel type
    city_mean = city.mean()
    resort_mean = resort.mean()

    # Display the mean ADR values
    st.write(f"City hotel mean ADR: {city_mean:.2f}")
    st.write(f"Resort hotel mean ADR: {resort_mean:.2f}")

    # Display the t-test results
    st.write(f"t-statistic: {t_stat:.2f}")
    st.write(f"P-value: {p_value:.4f}")

    # Determine the hypothesis test outcome
    alpha = 0.05
    if p_value < alpha:
        st.write("**Result:** Reject the null hypothesis")
        st.write("There is a significant difference in ADR between the City Hotel and Resort Hotel.")
    else:
        st.write("**Result:** Fail to reject the null hypothesis")
        st.write("There is no significant difference in ADR between the City Hotel and Resort Hotel.")

    st.write()
    st.write(f"The t-statistic is very large ({t_stat:.2f}), indicating a substantial difference in ADR.")
    st.write("The City Hotel has a significantly higher ADR compared to the Resort Hotel.")
    st.write("XYZ should focus on their City Hotel, as it generates more revenue.")

    st.write('# 4. `is_canceled` ❌')
    st.write("`is_canceled` column explains wheter or not a booking was cancelled by customers.")
    st.write("## 4.1. What are the features that minght affect whether or not a bookings is canceled?")

    # Correlation Check
    fig, ax = plt.subplots(figsize=(12, 8))

    mask = np.triu(np.ones_like(df.corr()))
    sns.heatmap(df.corr(), annot=True, cmap="Reds", mask=mask, linewidth=0.5, fmt=".2f")

    # Show the plot
    st.title("Correlation Heatmap")
    st.pyplot(fig)

if __name__ == '__main__':
    run()
