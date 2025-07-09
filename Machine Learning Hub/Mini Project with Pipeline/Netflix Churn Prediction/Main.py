import gradio as gr
import joblib
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')


def load_model() :
    model = joblib.load('model.pkl')
    return model


def prediction(age, gender, subscription_type, watch_hours, last_login_days,	
           region, device, monthly_fee, payment_method, number_of_profiles, avg_watch_time_per_day, favorite_genre):

    columns = [	'age',	'gender',	'subscription_type',	'watch_hours',	'last_login_days',	
           'region',	'device',	'monthly_fee',	'payment_method',	'number_of_profiles',	'avg_watch_time_per_day'	,'favorite_genre']
    
    input_data = pd.DataFrame([[age, gender, subscription_type, watch_hours, last_login_days,
                                region, device, monthly_fee, payment_method, number_of_profiles, avg_watch_time_per_day, favorite_genre]],
                              columns=columns)
    pipeline = load_model()
    prediction = pipeline.predict(input_data)
    return prediction[0]


demo = gr.Interface(
    fn=prediction,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(choices=["Male", "Female", "Other"], label="Gender"),
        gr.Dropdown(choices=["Basic", "Standard", "Premium"], label="Subscription Type"),
        gr.Number(label="Watch Hours (Monthly)"),
        gr.Number(label="Days Since Last Login"),
        gr.Dropdown(choices=["South America", "Asia", "Europe", "North America", "Africa", "Oceania"], label="Region"),
        gr.Dropdown(choices=["TV", "Mobile", "Laptop", "Tablet", "Desktop"], label="Device"),
        gr.Number(label="Monthly Fee ($)"),
        gr.Dropdown(choices=["Credit Card", "Debit Card", "Gift Card", "PayPal", "Crypto"], label="Payment Method"),
        gr.Number(label="Number of Profiles"),
        gr.Number(label="Avg Watch Time/Day (hrs)"),
        gr.Textbox(label="Favorite Genre"),
    ],
    outputs=[gr.Textbox(label='Prediction')],
    title="Netflix Churn Prediction",
    description="Fill in user and usage details to predict behavior or classification outcome."
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
