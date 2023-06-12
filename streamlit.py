import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import warnings

df = pd.read_csv("Crop_recommendation.csv")


loaded_model = pickle.load(open("model.pkl", 'rb'))

def predict_crop(N, P, K, temp, humidity, ph, rainfall):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = loaded_model.predict(single_pred)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return result
def eda(df):
    st.header("Exploratory Data Analysis")
    
    # Display the dataset
    st.subheader("Dataset")
    st.dataframe(df)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())
    # Pie chart for crop distribution
    st.subheader("Crop Distribution (Pie Chart)")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    crop_counts = df['label'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.pie(crop_counts.values, labels=crop_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.title('Crop Distribution')
    st.pyplot()
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    st.pyplot()
    #ScatterPlot
    st.subheader("Scatter Plot")
    sns.scatterplot(data=df, x='temperature', y='humidity', hue='label')
    plt.title("Temperature vs. Humidity")
    st.pyplot()
    #BoxPlot
    st.subheader("Box Plot for Numerical Variables")
    num_cols = ['temperature', 'humidity', 'ph', 'rainfall']
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[num_cols])
    plt.title("Distribution of Numerical Variables")
    st.pyplot()
    #PairPlot
    st.subheader("Pair Plot")
    sns.pairplot(data=df, hue='label')
    st.pyplot()
    



def main():
    st.set_page_config(page_title="Crop Prediction App", page_icon=":leaves:")
    st.markdown("<h1 style='text-align: center;'>CROP RECOMMENDATION</h1>", unsafe_allow_html=True)
    st.sidebar.title('Crop Yield Prediction')
    st.sidebar.subheader('Options')
    option = st.sidebar.selectbox('Select an option', ('Data Exploration','Prediction' ))

    # If the user selects 'Prediction'
    if option == 'Prediction':
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("Input parameters")
            N = st.number_input('Nitrogen', min_value=0, max_value=100,value=50)
            P = st.number_input('Phosphorus',min_value=0, max_value=100, value=40)
            K = st.number_input('Potassium',min_value=0, max_value=100, value=30)
            temp = st.slider('Temperature', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
            humidity = st.slider('Humidity', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            ph = st.slider('pH', min_value=0.0, max_value=14.0, value=7.0, step=0.1)
            rainfall = st.number_input('Rainfall (in mm)', value=100)
        with col2:
            st.image('https://agrisecure.com/wp-content/uploads/2020/02/shutterstock_1089153881-768x512.jpg', use_column_width=True)
            if st.button('Predict'):
                result = predict_crop(N, P, K, temp, humidity, ph, rainfall)
                st.success(result)
    elif option == 'Data Exploration':
        st.image('https://editor.analyticsvidhya.com/uploads/96804AI_Image_7.PNG')
        st.markdown("<h2 style='text-align: center;'>EXPLORATORY DATA ANALYSIS</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Here is the EDA for the crop recommendation dataset:</h4>", unsafe_allow_html=True)
        eda(df)


if __name__ == '__main__':
    main()
