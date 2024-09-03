import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

# Load models using context managers
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

stacking_model_N = load_model("stacking_model_N.pkl")
stacking_model_P = load_model("stacking_model_P.pkl")
stacking_model_K = load_model("stacking_model_K.pkl")

def calculate_deficiency(predicted, actual):
    return max(0, predicted - actual)

def predict_amount_of_fertilizer(N, P, K, temperature, humidity, ph, rainfall, label):
    input_data = [[temperature, humidity, ph, rainfall, label]]
    
    # Predictions
    predicted_N = stacking_model_N.predict(input_data)[0]
    predicted_P = stacking_model_P.predict(input_data)[0]
    predicted_K = stacking_model_K.predict(input_data)[0]
    
    # Calculate deficiencies
    deficient_N = calculate_deficiency(predicted_N, N)
    deficient_P = calculate_deficiency(predicted_P, P)
    deficient_K = calculate_deficiency(predicted_K, K)

    # Available fertilizers
    MOP, DAP, Urea = 0, 0, 0

    # Calculate recommendations
    if deficient_K > 0:
        MOP = deficient_K / 0.6

    if deficient_P > 0:
        DAP = deficient_P / 0.6
        remaining_deficient_N = max(0, deficient_N - (DAP * 0.18))
    else:
        remaining_deficient_N = deficient_N

    if remaining_deficient_N > 0:
        Urea = remaining_deficient_N / 0.6

    # Prepare recommendation message
    recommendation_message = []
    if MOP > 0:
        recommendation_message.append(f"{MOP:.2f} kgs of MOP")
    if DAP > 0:
        recommendation_message.append(f"{DAP:.2f} kgs of DAP")
    if Urea > 0:
        recommendation_message.append(f"{Urea:.2f} kgs of Urea")

    if recommendation_message:
        return "We recommend you add " + ", ".join(recommendation_message) + "."
    else:
        return "No fertilizer recommendation needed."

def main():
    st.title("Fertilizer Recommender")

    # Styling for Streamlit
    st.markdown("""
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Fertilizer Recommender ML App </h2>
    </div>
    """, unsafe_allow_html=True)

    # Input fields
    N_amount = st.number_input('N concentration (mg/L) of the soil', min_value=0.0, step=0.01, format="%.2f")
    P_amount = st.number_input('P concentration (mg/L) of the soil', min_value=0.0, step=0.01, format="%.2f")
    K_amount = st.number_input('K concentration (mg/L) of the soil', min_value=0.0, step=0.01, format="%.2f")
    temperature = st.number_input('Soil temperature (°C)', min_value=0.0, step=0.01, format="%.2f")
    humidity = st.number_input('Humidity of the air (%)', min_value=0.0, max_value=100.0, step=0.0001, format="%.4f")
    ph = st.number_input('pH Level', min_value=0.0, step=0.000000000000001, format="%.15f")
    rainfall = st.number_input('Amount of Rainfall/Irrigation (mm)', min_value=0.00, step=0.01, format="%.2f")
    
    # Crop types
    my_list = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
    crop_dict = {crop: idx for idx, crop in enumerate(my_list)}
    
    label = st.selectbox('Type of Crop', my_list)
    crop_type = crop_dict[label]

    result = ""
    if st.button("Predict"):
        result = predict_amount_of_fertilizer(N_amount, P_amount, K_amount, temperature, humidity, ph, rainfall, crop_type)
    
    st.success(result)

    if st.button("About"):
        st.text("This is a Machine Learning project made in the")
        st.text("Ethiopian Artificial Intelligence Institute Summer Camp of 2024.")

if __name__ == '__main__':
    main()
