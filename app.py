import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load Model 
model = tf.keras.models.load_model('model.keras')

# Load encoders and scaler
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geography.pkl", "rb") as file:
    onehot_encoder_geography = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit App Title
st.title("Customer Churn Prediction App 🚀")

# Sidebar for Inputs
st.sidebar.header("Enter Customer Details")

geography = st.sidebar.selectbox("Geography", onehot_encoder_geography.categories_[0])
gender = st.sidebar.selectbox("Gender", label_encoder_gender.classes_)
age = st.sidebar.slider("Age", 18, 92, 30)
balance = st.sidebar.number_input("Balance", min_value=0.0, step=100.0)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, step=1)
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=100.0)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_cr_card = st.sidebar.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member?", [0, 1])

# Submit Button
if st.sidebar.button("Predict Churn 🚀"):
    # Prepare input data
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # One-hot encode geography
    onehot_encoded = onehot_encoder_geography.transform(np.array([geography]).reshape(-1, 1)).toarray()
    encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder_geography.get_feature_names_out(["Geography"]))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), encoded_df], axis=1)

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    # Display churn probability with a progress bar
    st.subheader("Churn Probability")
    st.progress(float(prediction_proba))

    st.write(f"**Churn Probability:** {prediction_proba:.2%}")

    if prediction_proba > 0.5:
        st.error('🚨 The customer is likely to churn.')
    else:
        st.success('✅ The customer is not likely to churn.')
