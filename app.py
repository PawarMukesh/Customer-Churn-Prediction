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

st.write("Fill in the customer details below and click **Predict Churn** to get the result.")

# User Input Fields
geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92, 30)
balance = st.number_input("Balance", min_value=0.0, step=100.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=100.0)
tenure = st.slider("Tenure", 0, 10, 5)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])

# Submit Button
if st.button("Predict Churn 🚀"):
    try:
        # Encode Gender
        gender_encoded = label_encoder_gender.transform([gender])[0]

        # Prepare input data
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [gender_encoded],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        })

        # One-hot encode Geography
        onehot_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
        encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder_geography.get_feature_names_out(["Geography"]))

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), encoded_df], axis=1)

        # Scale data
        input_scaled = scaler.transform(input_data)

        # Predict churn
        prediction = model.predict(input_scaled)
        prediction_proba = prediction[0][0]

        # Display churn probability
        st.subheader("Churn Prediction Result")
        st.write(f"**Churn Probability:** {prediction_proba:.2%}")

        # Show result with colors
        if prediction_proba > 0.5:
            st.error('🚨 The customer is likely to churn.')
        else:
            st.success('✅ The customer is not likely to churn.')

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
