import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="PSstark/Machine-Learning-Prediction", filename="best_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Product Purchase Prediction App")
st.write("""
Welcome to the **Tourism Product Purchase Prediction App**! 🌍✨

This tool predicts whether a customer is likely to purchase a tourism product based on their personal details, preferences, and interaction history.

Please provide the customer information below, and the model will estimate the likelihood of them taking the product.
""")

# Basic demographic info
age = st.number_input("Customer Age", min_value=18, max_value=80, value=35)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# Contact and occupation info
typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])

# Travel and product preferences
city_tier = st.selectbox("City Tier", [1, 2, 3])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Numeric customer interaction details
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=100.0, value=10.0)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2)
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
number_of_trips = st.number_input("Number of Trips Taken", min_value=0, max_value=50, value=5)
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

# Additional info
passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1,2,3])
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=25000.0)

# 📊 Assemble all inputs into a DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# 🔮 Make prediction
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "✅ Customer is Likely to Purchase the Product" if prediction == 1 else "❌ Customer is Unlikely to Purchase the Product"
    st.subheader("Prediction Result:")
    st.success(result)
