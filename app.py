import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle
import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Load data and models once
@st.cache_resource
def load_data_and_models():
    df = pd.read_csv('Life_Expectancy.csv')
    
    # One-hot encode 'Country' and 'Status' columns
    df_encoded = pd.get_dummies(df, columns=['Country', 'Status'], drop_first=True)
    
    # Fill missing values with mean
    df_encoded.fillna(df_encoded.mean(), inplace=True)
    
    # Split features and target variable
    X = df_encoded.drop(['Life expectancy '], axis=1)
    y = df_encoded['Life expectancy ']
    
    # Split data into training and testing sets
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load models and one-hot encoding columns
    models = {}
    with open('Random_forest_model.pkl', 'rb') as f:
        models['Random Forest'] = pickle.load(f)
    with open('decision_tree_model.pkl', 'rb') as f:
        models['Decision Tree'] = pickle.load(f)
    with open('extra_trees_model.pkl', 'rb') as f:
        models['Extra Trees'] = pickle.load(f)
    with open('linear_regression_model.pkl', 'rb') as f:
        models['Linear Regression'] = pickle.load(f)
    with open('one_hot_columns.pkl', 'rb') as f:
        one_hot_columns = pickle.load(f)
    
    return df, df_encoded, xte, yte, models, one_hot_columns

df, df_encoded, xte, yte, models, one_hot_columns = load_data_and_models()

# Precompute model accuracies
accuracies = {}
for model_name, model in models.items():
    y_pred = model.predict(xte)
    r2 = r2_score(yte, y_pred)
    mse = mean_squared_error(yte, y_pred)
    accuracies[model_name] = (r2, mse)


# Define the navigation menu
selected = option_menu(None, ["Home", "About", "Dashboard", "Prediction"], 
                       icons=['house', 'info-circle', 'speedometer', 'bar-chart'], 
                       menu_icon="cast", default_index=0, orientation="horizontal")



st.markdown("""
<div style="text-align:center; font-size:32px; font-weight:bold;">
Life Expectancy Prediction Using Machine Learning Models
</div>
""", unsafe_allow_html=True)


# Define content for each page
def home_page():

    st.image("image.png", caption="Welcome to the Life Expectancy Prediction App")
    st.write("""
        This application helps to predict the life expectancy of population based on various health and
        socio-economic factors. Navigate through the app using the menu above to learn more about the
        project, view the dashboard, or make predictions.
    """)

def about_page():
    st.header("About")
    st.markdown("""
        Welcome to the Life Expectancy Prediction Application. This app leverages advanced machine learning algorithms to predict life expectancy based on a diverse range of health and socio-economic factors. By exploring this app, you can gain insights into how various indicators influence life expectancy across different countries and over time.

        #### Project Objectives

        - **Insight Generation**: Identify key factors affecting life expectancy.
        - **Predictive Analysis**: Provide accurate life expectancy predictions using machine learning models.
        - **Interactive Visualization**: Enable users to explore and analyze data through interactive dashboards.

        #### Technologies Used

        - **Python**: The primary programming language used for data processing and machine learning.
        - **Streamlit**: A powerful framework for building interactive web applications with Python.
        - **Scikit-learn**: A robust machine learning library used for building and training predictive models.
        - **Pandas**: A versatile data manipulation library essential for data cleaning and preparation.
        - **Plotly & Seaborn**: Libraries for creating interactive and visually appealing plots and charts.

        #### Data Overview

        The dataset utilized in this project includes multiple indicators such as:

        - **Health Indicators**: Metrics like BMI, alcohol consumption, and incidence of diseases.
        - **Socio-economic Factors**: Variables including GDP, schooling, and population size.
        - **Mortality Rates**: Data on adult and infant mortality rates.

        #### Machine Learning Models

        To ensure high prediction accuracy, the following models have been trained and evaluated:

        - **Random Forest Regressor**
        - **Decision Tree Regressor**
        - **Extra Trees Regressor**
        - **Linear Regression**

        Each model has been fine-tuned and evaluated to determine its performance, providing you with reliable life expectancy predictions.

        #### How to Use This App

        - **Home**: Get an overview of the app and its capabilities.
        - **Dashboard**: Explore interactive visualizations and gain insights into the data.
        - **Prediction**: Input specific data to get predictions on life expectancy using various machine learning models.

        This application aims to empower users with the knowledge and tools needed to understand and predict life expectancy trends, ultimately contributing to better health planning and resource allocation.

        Thank you for using the Life Expectancy Prediction App. We hope you find it insightful and useful.
    """)


def dashboard_page():

    st.title("Life Expectancy Dashboard")
    st.write("""
        This dashboard provides various metrics and visualizations related to life expectancy data. 
        Explore the charts and graphs below to gain insights into the factors affecting life expectancy.
    """)

    # Load the dataset
    URL_DATASET = 'Life_Expectancy.csv'
    df = pd.read_csv(URL_DATASET)

    # Sidebar Filters
    st.sidebar.title("Filters")
    choice = st.sidebar.radio("Choice", ["By default", "Comparison"], index=0)

    if choice == "By default":
        selected_countries = df["Country"].unique()
        selected_years = (df["Year"].min(), df["Year"].max())
        st.sidebar.write("All countries are selected by default.")
        graph_title_prefix = "Overall"
    else:
        selected_countries = st.sidebar.multiselect(
            "Select Countries",
            options=df["Country"].unique(),
            default=[]
        )
        selected_years = st.sidebar.slider(
            "Select Years",
            min_value=int(df["Year"].min()),
            max_value=int(df["Year"].max()),
            value=(int(df["Year"].min()), int(df["Year"].max())),
            key="year_slider"
        )
        graph_title_prefix = ", ".join(selected_countries)
    # Precompute ISO codes for all unique countries
    country_iso_dict = {}
    for country in df['Country'].unique():
        try:
            country_iso_dict[country] = pycountry.countries.lookup(country).alpha_3
        except:
            country_iso_dict[country] = 'Unknown'

    df['iso_alpha'] = df['Country'].map(country_iso_dict)
        # Apply the filters
    filtered_df = df[(df["Year"].between(*selected_years)) & (df["Country"].isin(selected_countries))]

    # Layout the dashboard
    col1, col2 = st.columns(2)
    with col1:
        # Top Maximum Life Expectancy Chart
        top_max_life_expectancy = filtered_df.groupby("Country", as_index=False)["Life expectancy "].mean().nlargest(20, "Life expectancy ")
        fig_max_life_expectancy = px.bar(top_max_life_expectancy, x="Country", y="Life expectancy ", title=f"{graph_title_prefix} - Top Maximum Life Expectancy")
        fig_max_life_expectancy.update_layout(xaxis={'categoryorder':'total ascending', 'tickangle':45})
        st.plotly_chart(fig_max_life_expectancy, use_container_width=True)

    with col2:
        # Top Minimum Life Expectancy Chart
        top_min_life_expectancy = filtered_df.groupby("Country", as_index=False)["Life expectancy "].mean().nsmallest(20, "Life expectancy ")
        fig_min_life_expectancy = px.bar(top_min_life_expectancy, x="Country", y="Life expectancy ", title=f"{graph_title_prefix} - Top Minimum Life Expectancy")
        fig_min_life_expectancy.update_layout(xaxis={'categoryorder':'total ascending', 'tickangle':45})
        st.plotly_chart(fig_min_life_expectancy, use_container_width=True)

    # Correlation with Life Expectancy
    st.subheader("Top 5 feature Importance based on Decision Tree Correlation with Life Expectancy")
    col1, col2, col3, col4, col5 = st.columns(5)
    selected_factors = [" HIV/AIDS","Adult Mortality", 'Income composition of resources', 'Schooling', ' BMI ']

    for idx, factor in enumerate(selected_factors):
        if factor in filtered_df.columns:
            correlation = filtered_df["Life expectancy "].corr(filtered_df[factor])
            with [col1, col2, col3, col4, col5][idx]:
                st.metric(label=f"{factor.strip()}", value=f"{correlation:.2f}")
        else:
            with [col1, col2, col3, col4, col5][idx]:
                st.metric(label=f"{factor.strip()}", value="N/A")

    # Developed VS Developing Pie Chart
    col1, col2 = st.columns(2)
    with col1:
        status_count = filtered_df["Status"].value_counts()
        fig_status_pie = px.pie(names=status_count.index, values=status_count.values, title=f"{graph_title_prefix} - Developed VS Developing Countries")
        st.plotly_chart(fig_status_pie, use_container_width=True)

    with col2:
        # Trend of Life Expectancy Over the Years Line Chart
        top_countries = filtered_df.groupby("Country")["Life expectancy "].mean().nlargest(5).index.tolist()
        filtered_top_df = filtered_df[(filtered_df["Country"].isin(top_countries)) & (filtered_df["Year"].between(selected_years[0], selected_years[1]))]

        fig_trend_life_expectancy = px.line(filtered_top_df, x='Year', y='Life expectancy ', color='Country', title=f"{graph_title_prefix} - Trend of Life Expectancy Over the Years")
        st.plotly_chart(fig_trend_life_expectancy, use_container_width=True)

       
    # Average Life Expectancy Over the Years
    col1, col2 = st.columns(2)
    with col1:
        avg_life_expectancy = filtered_df.groupby("Year")["Life expectancy "].mean()
        fig_avg_life_expectancy = px.scatter(x=avg_life_expectancy.index, y=avg_life_expectancy.values, trendline="lowess")
        fig_avg_life_expectancy.update_layout(
            title=f"{graph_title_prefix} - Average Life Expectancy Over the Years",
            xaxis_title="Year",
            yaxis_title="Average Life Expectancy"
        )
        st.plotly_chart(fig_avg_life_expectancy, use_container_width=True)
    with col2:
        # Scatter Plot Matrix
        fig_scatter_matrix = plt.figure(figsize=(10, 10))
        sns.scatterplot(data=filtered_df[["Life expectancy ", "Adult Mortality", "infant deaths", "Hepatitis B", ' HIV/AIDS', "Polio"]], x="Life expectancy ", y="Adult Mortality", hue='Polio')
        st.pyplot(fig_scatter_matrix)

    # Choropleth Map and Heatmap
    st.subheader(f"{graph_title_prefix} - Life Expectancy of Various Countries Over the Years")

    # Choropleth Map
    fig_choropleth = px.choropleth(
        filtered_df,
        locations="iso_alpha",
        color="Life expectancy ",
        hover_name="Country",
        color_continuous_scale=px.colors.sequential.Plasma,
        animation_frame="Year",
        title="Life Expectancy of Various Countries Over the Years"
    )
    st.plotly_chart(fig_choropleth) 

    # Heatmap
    fig_heatmap = plt.figure(figsize=(8, 8))
    sns.heatmap(filtered_df[["Life expectancy ", "Adult Mortality", "infant deaths", "Hepatitis B", " HIV/AIDS", "Polio"]].corr(), annot=True, cmap="YlOrRd")
    plt.title(f"{graph_title_prefix} - Correlation Heatmap")
    st.pyplot(fig_heatmap)


    # Bubble Chart
    st.subheader(f"{graph_title_prefix} - Life Expectancy, GDP, and Population")
    fig_bubble = px.scatter(
        filtered_df,
        x="GDP",
        y="Life expectancy ",
        size="Population",
        color="Country",
        hover_name="Country",
        title=f"{graph_title_prefix} - Life Expectancy, GDP, and Population"
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    # Tree Map
    st.subheader(f"{graph_title_prefix} - Life Expectancy by Country")
    fig_treemap = px.treemap(
        filtered_df,
        path=["Country", "Life expectancy "],
        values="Life expectancy ",
        title=f"{graph_title_prefix} - Life Expectancy by Country"
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

    # Download original Dataset
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Data', data=csv, file_name="Life_Expectancy_Data.csv", mime="text/csv")

def prediction_page():
    st.header("Prediction Page")
    st.write("""
        Use this page to make predictions about life expectancy based on input data. 
        Enter the required information in the form below and click the 'Predict' button to see the results.
    """)
    
    st.title("Life Expectancy Predictor")

    # Sample values for population, GDP, and infant deaths
    sample_population = 12753375.12
    sample_gdp = 7483.158
    sample_infant_deaths = 30

    # CSS styling
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f0f0;
            color: #333;
        }
        .sidebar .sidebar-content .block-container {
            padding: 10px;
        }
        .sidebar .sidebar-content .block-container h1 {
            color: #ff6347;
        }
        .stTextInput>div>div>input {
            background-color: #f0f0f0 !important;
            color: #333 !important;
        }
        .stButton>button {
            background-color: #ff6347 !important;
            color: white !important;
            border: 2px solid #ff6347 !important;
        }
        .stButton>button:hover {
            background-color: #ff6347 !important;
            color: white !important;
        }
        .stSlider>div>div>div>div>div>div>input {
            background-color: #f0f0f0 !important;
            color: #333 !important.
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Input elements for user input
    with st.sidebar:
        st.title("Select Model")
        model_type = st.selectbox('Choose Model', ['Random Forest', 'Decision Tree', 'Extra Trees', 'Linear Regression'])
        st.title("Input Features")
        st.subheader("Health Factors")
        country = st.selectbox('Country', df['Country'].unique())
        year = st.slider('Year', 2000, 2024, 2019)
        status = st.selectbox('Status', df['Status'].unique())
        adult_mortality = st.slider('Adult Mortality (per 1000)', min_value=0, max_value=1000, step=1)
        infant_deaths = st.slider('Infant Deaths', min_value=0, max_value=1800, step=1, value=sample_infant_deaths)
        alcohol = st.slider('Alcohol Consumption (liters)', min_value=0.0, max_value=100.0, step=0.1)
        percentage_expenditure = st.slider('Percentage Expenditure (%)', min_value=0.0, max_value=100.0, step=0.1)
        hepatitis_b = st.slider('Hepatitis B Coverage (%)', min_value=0.0, max_value=100.0, step=0.1)
        measles = st.slider('Measles Cases', min_value=0, max_value=10000, step=1)
        bmi = st.slider('BMI', min_value=0.0, max_value=100.0, step=0.1)
        under_five_deaths = st.slider('Under-Five Deaths', min_value=0, max_value=500, step=1)
        polio = st.slider('Polio Coverage (%)', min_value=0.0, max_value=100.0, step=0.1)
        st.subheader("Economic Factors")
        total_expenditure = st.slider('Total Expenditure (%)', min_value=0.0, max_value=100.0, step=0.1)
        diphtheria = st.slider('Diphtheria Coverage (%)', min_value=0.0, max_value=100.0, step=0.1)
        hiv_aids = st.slider('HIV/AIDS Cases', min_value=0.0, max_value=100.0, step=0.1)
        gdp = st.slider('GDP', min_value=0, max_value=120000, step=100, value=int(sample_gdp))
        population = st.slider('Population', min_value=0, max_value=1300000000, step=10000, value=int(sample_population))
        thinness_1_19_years = st.slider('Thinness 1-19 Years (%)', min_value=0.0, max_value=100.0, step=0.1)
        thinness_5_9_years = st.slider('Thinness 5-9 Years (%)', min_value=0.0, max_value=100.0, step=0.1)
        income_composition = st.slider('Income Composition of Resources', min_value=0.0, max_value=1.0, step=0.01)
        schooling = st.slider('Schooling', min_value=0.0, max_value=100.0, step=0.1)

    # Show model accuracy based on selected model
    if model_type in accuracies:
        r2, mse = accuracies[model_type]
        st.write(f"Model Accuracy ({model_type} - R2 Score): {r2:.2f}")
        st.write(f"Model Accuracy ({model_type} - Mean Squared Error): {mse:.2f}")

    # Add some empty space above and below the button
    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button
    if st.button('Predict Life Expectancy'):
        # Prepare input for prediction
        input_data = pd.DataFrame({
            'Country': [country],
            'Year': [year],
            'Status': [status],
            'Life expectancy ': [0],  # Dummy value for the target variable
            'Adult Mortality': [adult_mortality],
            'infant deaths': [infant_deaths],
            'Alcohol': [alcohol],
            'percentage expenditure': [percentage_expenditure],
            'Hepatitis B': [hepatitis_b],
            'Measles ': [measles],
            ' BMI ': [bmi],
            'under-five deaths ': [under_five_deaths],
            'Polio': [polio],
            'Total expenditure': [total_expenditure],
            'Diphtheria ': [diphtheria],
            ' HIV/AIDS': [hiv_aids],
            'GDP': [gdp],
            'Population': [population],
            ' thinness  1-19 years': [thinness_1_19_years],
            ' thinness 5-9 years': [thinness_5_9_years],
            'Income composition of resources': [income_composition],
            'Schooling': [schooling]
        })

        # Convert dataframe to native Python types
        input_data_native = input_data.astype({
            'Adult Mortality': float,
            'infant deaths': int,
            'Alcohol': float,
            'percentage expenditure': float,
            'Hepatitis B': float,
            'Measles ': int,
            ' BMI ': float,
            'under-five deaths ': int,
            'Polio': float,
            'Total expenditure': float,
            'Diphtheria ': float,
            ' HIV/AIDS': float,
            'GDP': float,
            'Population': int,
            ' thinness  1-19 years': float,
            ' thinness 5-9 years': float,
            'Income composition of resources': float,
            'Schooling': float
        })

        # Handle one-hot encoding
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(df[['Country', 'Status']])
        country_encoded = encoder.transform([[country, status]]).toarray()
        country_columns = encoder.get_feature_names_out(['Country', 'Status'])

        # Create a DataFrame with encoded columns
        encoded_df = pd.DataFrame(country_encoded, columns=country_columns)

        # Concatenate the encoded columns with the rest of the input data
        input_data_native = pd.concat([input_data_native.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Ensure all columns in the one-hot encoding are present
        for col in one_hot_columns:
            if col not in input_data_native:
                input_data_native[col] = 0

        # Select only the columns present in the training data
        input_data_native = input_data_native[one_hot_columns]

        # Predict using the selected model
        model = models[model_type]
        prediction = model.predict(input_data_native)
        st.write(f"Predicted Life Expectancy: {prediction[0]:.2f}")

# Run the app
# Render the selected page
if selected == "Home":
    home_page()
elif selected == "About":
    about_page()
elif selected == "Dashboard":
    dashboard_page()
elif selected == "Prediction":
    prediction_page()

# To run the app, use the command: streamlit run app.py
