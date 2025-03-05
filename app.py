import streamlit as st
import pickle
import numpy as np

# Load trained models and MultiLabelBinarizer
with open("skill_prediction_models.pkl", "rb") as model_file:
    models = pickle.load(model_file)

with open("skill_mlb.pkl", "rb") as mlb_file:
    mlb = pickle.load(mlb_file)

# Streamlit UI
st.title("🔮 Skill Prediction App")
st.write("Enter the skills you already have, and we'll predict what you should learn next!")

# User input
user_input = st.text_area("Enter skills (comma-separated):")

if st.button("Predict Next Skills"):
    if not user_input:
        st.warning("Please enter at least one skill!")
    else:
        # Process input skills
        input_skills = [skill.strip().lower() for skill in user_input.split(",")]

        # Create an empty encoded array with the same shape as model input
        input_encoded = np.zeros((1, len(mlb.classes_)))  

        # Encode user input using the trained MLB
        for skill in input_skills:
            if skill in mlb.classes_:
                idx = np.where(mlb.classes_ == skill)[0][0]  # Find skill index
                input_encoded[0, idx] = 1  # Set corresponding index to 1

        predicted_skills = []

        # Check feature consistency before prediction
        model_feature_count = list(models.values())[0].n_features_in_
        if input_encoded.shape[1] != model_feature_count:
            st.error(f"⚠️ Feature mismatch: Model expects {model_feature_count}, but input has {input_encoded.shape[1]}. Try retraining with the same MLB.")
        else:
            # Predict missing skills
            for skill, model in models.items():
                if skill not in input_skills:  # Predict only for missing skills
                    prediction = model.predict(input_encoded)  
                    if prediction[0] == 1:
                        predicted_skills.append(skill)

            # Display results
            if predicted_skills:
                st.success("📚 You should learn these skills next:")
                st.write(", ".join(predicted_skills))
            else:
                st.info("No new skills recommended based on your input.")
