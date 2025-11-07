import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle 

# ---- Load model and encoders ----
model = tf.keras.models.load_model('regression_model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---- Streamlit UI ----
st.title('💰 Customer Salary Prediction')

st.markdown("This app predicts a customer’s **Estimated Salary** based on their demographic and financial information.")

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# ---- Prepare input ----
input_data = {
    "CreditScore": [credit_score],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Geography": [geography]
}

input_data = pd.DataFrame(input_data)

# Encode categorical features
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Merge all features
input_data = pd.concat([input_data.drop(['Geography'], axis=1).reset_index(drop=True), geo_encoded_df], axis=1)

# ✅ Ensure same column order as during training
expected_columns = scaler.feature_names_in_  # works if you trained scaler in sklearn >=1.0
missing_cols = [col for col in expected_columns if col not in input_data.columns]

# Add any missing columns with 0s
for col in missing_cols:
    input_data[col] = 0


# Scale the input
input_data_scaled = scaler.transform(input_data)

# ---- Predict ----
predicted_salary = model.predict(input_data_scaled)[0][0]

# ---- Display Result ----
st.subheader("💼 Predicted Estimated Salary:")

st.success(f"${predicted_salary:,.2f}")
