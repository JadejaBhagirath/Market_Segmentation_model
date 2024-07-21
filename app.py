import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = pickle.load(open('gb_model.sav', 'rb'))

# Custom CSS to change the background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: lightblue;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.title('Customer Segmentation Model Deployment')

# Input fields for the features
st.header('Input Customer Data')

balance = st.number_input('Balance')
balance_frequency = st.number_input('Balance Frequency')
purchases = st.number_input('Purchases')
oneoff_purchases = st.number_input('One-off Purchases')
installments_purchases = st.number_input('Installments Purchases')
cash_advance = st.number_input('Cash Advance')
purchases_frequency = st.number_input('Purchases Frequency')
oneoff_purchases_frequency = st.number_input('One-off Purchases Frequency')
purchases_installments_frequency = st.number_input('Purchases Installments Frequency')
cash_advance_frequency = st.number_input('Cash Advance Frequency')
cash_advance_trx = st.number_input('Cash Advance Transactions')
purchases_trx = st.number_input('Purchases Transactions')
credit_limit = st.number_input('Credit Limit')
payments = st.number_input('Payments')
minimum_payments = st.number_input('Minimum Payments')
prc_full_payment = st.number_input('Percentage Full Payment')
tenure = st.number_input('Tenure')

# Prepare the input data
input_data = pd.DataFrame({
    'BALANCE': [balance],
    'BALANCE_FREQUENCY': [balance_frequency],
    'PURCHASES': [purchases],
    'ONEOFF_PURCHASES': [oneoff_purchases],
    'INSTALLMENTS_PURCHASES': [installments_purchases],
    'CASH_ADVANCE': [cash_advance],
    'PURCHASES_FREQUENCY': [purchases_frequency],
    'ONEOFF_PURCHASES_FREQUENCY': [oneoff_purchases_frequency],
    'PURCHASES_INSTALLMENTS_FREQUENCY': [purchases_installments_frequency],
    'CASH_ADVANCE_FREQUENCY': [cash_advance_frequency],
    'CASH_ADVANCE_TRX': [cash_advance_trx],
    'PURCHASES_TRX': [purchases_trx],
    'CREDIT_LIMIT': [credit_limit],
    'PAYMENTS': [payments],
    'MINIMUM_PAYMENTS': [minimum_payments],
    'PRC_FULL_PAYMENT': [prc_full_payment],
    'TENURE': [tenure]
})

# Scale the input data
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Predict the cluster
if st.button('Predict Cluster'):
    prediction = model.predict(input_data_scaled)
    st.write(f'The customer belongs to cluster: {prediction[0]}')
