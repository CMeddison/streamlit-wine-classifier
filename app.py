import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_wine


model_scaler= pickle.load(open('models/wine_scaler.pkl','rb'))
model_lr= pickle.load(open('models/wine_log_reg.pkl','rb'))
model_dt= pickle.load(open('models/wine_dtc.pkl','rb'))

#load wine dataset
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_classes = wine.target_names
#App title
st.title("App for classification of wine")
st.subheader("Choose a model and input feature values to predict wine class")
models={"Logistic Regression":model_lr,"Decision Tree":model_dt}
#user select model
selected_model=st.selectbox("Select a model",list(models.keys()))
chosen_model=models[selected_model]
input_data={}
for col in wine_df.columns:
    input_data[col] = st.slider(col, min_value=wine_df[col].min(), max_value=wine_df[col].max())
# convert dict to df
input_df = pd.DataFrame([input_data])
# Scale the input data if Logistic Regression is selected
if selected_model == "Logistic Regression":
    input_df_scaled = model_scaler.transform(input_df)
else:
    input_df_scaled = input_df  # No scaling needed for Decision Tree
if st.button("Predict"):
    predicted = model_lr.predict(input_df_scaled)[0]
    predicted_prob=model_lr.predict_proba(input_df_scaled)[0]
    # display prediction
    st.subheader("Prediction Results")
if predicted == 0:
        st.write("The wine is predicted to be from Class 1")
elif predicted==1:
        st.write("The wine is predicted to be from Class 2")
else:
        st.write("The wine is predicted to be from Class 3")
        st.write("**Class Probabilities:**")
probabilities_df = pd.DataFrame({
        "Class": wine_classes,
        "Probability": predicted_prob
    }).sort_values(by="Probability", ascending=False)
st.write(probabilities_df)
