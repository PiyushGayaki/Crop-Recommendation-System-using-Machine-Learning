# Crop-Recommendation-System-using-Machine-Learning
A crop recommendation system based on machine learning can help farmers make accurate choices about crop production, resulting in better yields, decreased resource waste, and higher profit. It can revolutionize Indian agriculture by promoting sustainable farming methods.

## Dataset
The dataset was gathered from Kaggle and includes agricultural features such as nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall, and crop label. It has 8 attributes and 2200 samples.
Pre-processing the data is used to transform it into usable and effective information. Feature selection is used to find the most relevant traits, while data splitting is used to calculate the model's performance on test data. 70% of the data is used for training and 30% for testing.

## Model Selection
Nine machine learning models are then trained on the dataset, and their accuracy scores are printed.
Random Forest is used to predict the best crop based on inputs such as N, P, K, temperature, humidity, pH, and rainfall. The model is saved as a pickle file for further deployment. If the model is unable to make any predictions, the final else line will be executed.

## Deployment 
Streamlit is used for deployment by importing libraries and loading a pre-trained machine learning model. The eda() function is used for exploratory data analysis, and the predict_crop() method is used to predict the optimum crop. The app has two options: "Data Exploration" and "Prediction". The user can enter values and view results on the same page.

# Important NOTE
Research Paper will be Uploaded within few days, it will content all information about execution. 
