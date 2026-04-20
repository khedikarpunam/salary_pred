import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest model
model = joblib.load('best_model.pkl')

# --- Recreate Label Encoders for categorical features ---
# It's crucial to use the same mappings as during training.
# We will recreate and store them in a dictionary.

# Load the original dataset to fit LabelEncoders
# Assuming the original 'Salary_Dataset_DataScienceLovers.csv' is available
try:
    original_df = pd.read_csv('Salary_Dataset.csv')
except FileNotFoundError:
    st.error("Original dataset 'Salary_Dataset.csv' not found. Please ensure it's in the /content directory.")
    st.stop()

# Define the categorical columns that were encoded
categorical_cols = ['Rating', 'Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles']

# Initialize a dictionary to store fitted LabelEncoders
label_encoders = {}

# Fit a LabelEncoder for each categorical column using the original data
for col in categorical_cols:
    le = LabelEncoder()
    original_df[col] = original_df[col].astype(str) # Ensure consistent type for LabelEncoder
    le.fit(original_df[col].unique())
    label_encoders[col] = le

# --- Streamlit Application Layout ---
st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields for user
input_data = {}

for col in categorical_cols:
    options = original_df[col].unique().tolist()
    options.sort() # Sort options for better UI
    selected_value = st.selectbox(f'Select {col.replace('_', ' ')}', options)
    input_data[col] = selected_value

input_data['Salaries Reported'] = st.number_input('Number of Salaries Reported', min_value=1, value=1)


if st.button('Predict Salary'):
    processed_input = {}
    for col in categorical_cols:
        processed_input[col] = label_encoders[col].transform([input_data[col]])[0]
    
    processed_input['Salaries Reported'] = input_data['Salaries Reported']

    feature_order = ['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']
    input_df = pd.DataFrame([processed_input], columns=feature_order)
    
    prediction = model.predict(input_df)
    
    st.success(f'Predicted Salary: {prediction[0]:,.2f}')
