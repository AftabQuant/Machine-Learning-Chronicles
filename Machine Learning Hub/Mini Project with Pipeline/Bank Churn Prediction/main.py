import gradio as gr
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load trained model
def load_model():
    model = joblib.load('model.pkl')
    return model

# Prediction function
def prediction(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts,
               HasCrCard, IsActiveMember, EstimatedSalary):

    # Convert categorical Yes/No to binary values
    HasCrCard = 1 if HasCrCard == "Yes" else 0
    IsActiveMember = 1 if IsActiveMember == "Yes" else 0

    # Input as DataFrame
    columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
               'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    input_data = pd.DataFrame([[CreditScore, Geography, Gender, Age, Tenure, Balance,
                                 NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]],
                              columns=columns)
    
    # Load and predict
    model = load_model()
    prediction = model.predict(input_data)

    return "Exited" if prediction[0] == 1 else "Retained"

# Gradio Interface
demo = gr.Interface(
    fn=prediction,
    inputs=[
        gr.Number(label="Credit Score"),
        gr.Radio(choices=["France", "Germany", "Spain"], label="Geography"),
        gr.Radio(choices=["Male", "Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Dropdown(choices=list(range(0, 11)), label="Tenure"),
        gr.Number(label="Balance"),
        gr.Radio(choices=[1, 2, 3, 4], label="Number of Products"),
        gr.Radio(choices=["Yes", "No"], label="Has Credit Card"),
        gr.Radio(choices=["Yes", "No"], label="Is Active Member"),
        gr.Number(label="Estimated Salary"),
    ],
    outputs=gr.Textbox(label='Prediction'),
    title="Customer Churn Prediction",
    description="Fill in user details to predict whether the customer is likely to churn."
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
