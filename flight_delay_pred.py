import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import base64
from datetime import datetime
import xgboost as xgb  # Import XGBoost

# Load LabelEncoder and StandardScaler
label_encoders = joblib.load(os.path.join('JOBLIB', 'label_encoder_complete.joblib'))
scaler = joblib.load(os.path.join('JOBLIB', 'scalar_complete.joblib'))

# Define model file paths
model_files = {
    'Logistic Regression': os.path.join('JOBLIB', 'logistic_regression_model_complete.joblib'),
    'XGBoost': os.path.join('JOBLIB', 'xgboost_model_complete.joblib')  # Update with your XGBoost model path
}

# Columns to encode and all columns used for scaling
columns_to_encode = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'WEEK']
all_columns = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'WEEK', 
                'DEPARTURE_DELAY', 'TAXI_IN', 'TAXI_OUT', 'WHEELS_OFF',
                'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME']

# Convert hours and minutes to total minutes
def convert_to_minutes(hour, minute):
    return hour * 60 + minute

# Function to apply color styling based on the value
def apply_color_style(val, prediction_type):
    if prediction_type == 'classification':
        if isinstance(val, str):
            return 'color: red' if val == 'Delayed' else 'color: white'
        return ''
    elif prediction_type == 'regression':
        if isinstance(val, (int, float)):
            return 'color: red' if val > 0 else 'color: white'
        return ''
    return ''

# Helper function to encode image file to base64
def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Streamlit app
def main():
    st.set_page_config(page_title="Flight Delay Prediction", page_icon=":airplane_departure:")

    st.title('🛫 Flight Delay Prediction App')

    # Load local image and encode it to base64
    image_path = 'bg-image/bg.jpg'  # Update with your local file path
    encoded_image = load_image(image_path)
    
    # Add background image to the Streamlit app
    st.markdown(
        f"""
        <style>
        body {{
            background-image: url(data:image/png;base64,{encoded_image}); 
            background-size: cover; 
            background-position: center;
            color: #333333;  
        }}
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_image});
            background-size: cover;
            background-position: center;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )

    # Option to choose between file upload and manual input
    option = st.selectbox('Choose Input Method', ['', 'Upload Test Data', 'Manual Input'])

    # Load models
    logistic_model = joblib.load(model_files['Logistic Regression'])
    xgboost_model = joblib.load(model_files['XGBoost'])  # Load the XGBoost model

    if option == 'Upload Test Data':
        new_data_file = st.file_uploader("Upload new dataset (CSV)", type=["csv"])

        if new_data_file:
            new_data = pd.read_csv(new_data_file)
            st.write("Uploaded Data:", new_data)

            # Prepare data for predictions
            result_columns = new_data[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DATE']].copy()
            new_data = new_data.drop(['DATE'], axis=1, errors='ignore')

            # Encode categorical features
            for col in columns_to_encode:
                if col in new_data.columns:
                    encoder = label_encoders[col]
                    new_data[col] = encoder.transform(new_data[col])

            # Ensure all required columns are present and in the correct order
            new_data = new_data.reindex(columns=all_columns, fill_value=0)

            # Scale the data
            new_data_scaled = scaler.transform(new_data)
            new_data_scaled = pd.DataFrame(new_data_scaled, columns=all_columns)

            # Make predictions
            logistic_predictions = logistic_model.predict(new_data_scaled)
            logistic_predictions = np.where(logistic_predictions == 1, 'Delayed', 'Not Delayed')
            xgboost_predictions = xgboost_model.predict(new_data_scaled)

            # Prepare results
            results = result_columns.copy()
            results.insert(0, 'IS_DELAYED', logistic_predictions)
            results.insert(1, 'ARRIVAL_DELAY', xgboost_predictions)

            st.write("Results:")
            # Apply color styling to only the prediction columns
            styled_results = results.style.applymap(lambda v: apply_color_style(v, 'classification') if 'IS_DELAYED' in results.columns and v in results['IS_DELAYED'].values else '')

            # Apply color styling to only the ARRIVAL_DELAY column
            styled_results = styled_results.applymap(lambda v: apply_color_style(v, 'regression') if 'ARRIVAL_DELAY' in results.columns and v in results['ARRIVAL_DELAY'].values else '')

            st.dataframe(styled_results)

            # Option to download results as CSV
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='results.csv',
                mime='text/csv',
                key='download-csv'
            )
                
    elif option == 'Manual Input':
        st.subheader("Enter Flight Details")

        # Get unique values for select boxes from label encoders
        def get_unique_values(encoder):
            return [''] + list(encoder.classes_)

        airline_options = get_unique_values(label_encoders['AIRLINE'])
        origin_airport_options = get_unique_values(label_encoders['ORIGIN_AIRPORT'])
        destination_airport_options = get_unique_values(label_encoders['DESTINATION_AIRPORT'])
        week_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        airline = st.selectbox('Airline', airline_options)
        origin_airport = st.selectbox('Origin Airport', origin_airport_options)
        destination_airport = st.selectbox('Destination Airport', destination_airport_options)
        week = st.selectbox('Week', [''] + week_options)

        st.subheader("Enter Time Details")

        # Create columns for hour and minute inputs
        col1, col2, col3 = st.columns(3)

        with col1:
            scheduled_departure_hour = st.number_input('Scheduled Departure Hour', min_value=0, max_value=23, value=0)
            departure_hour = st.number_input('Departure Hour', min_value=0, max_value=23, value=0)
            wheels_off_hour = st.number_input('Wheels Off Hour', min_value=0, max_value=23, value=0)

        with col2:
            scheduled_departure_minute = st.number_input('Scheduled Departure Minute', min_value=0, max_value=59, value=0)
            departure_minute = st.number_input('Departure Minute', min_value=0, max_value=59, value=0)
            wheels_off_minute = st.number_input('Wheels Off Minute', min_value=0, max_value=59, value=0)

        with col3:
            taxi_out_time = st.number_input('Taxi Out Time (minutes)', min_value=0)
            taxi_in_time = st.number_input('Taxi In Time (minutes)', min_value=0)
            date = st.date_input('Date')

        if st.button('Predict'):
            try:
                scheduled_departure_minutes = convert_to_minutes(scheduled_departure_hour, scheduled_departure_minute)
                departure_time_minutes = convert_to_minutes(departure_hour, departure_minute)
                wheels_off_minutes = convert_to_minutes(wheels_off_hour, wheels_off_minute)

                # Calculate Departure Delay
                departure_delay = departure_time_minutes - scheduled_departure_minutes

                input_data = pd.DataFrame({
                    'AIRLINE': [airline],
                    'ORIGIN_AIRPORT': [origin_airport],
                    'DESTINATION_AIRPORT': [destination_airport],
                    'WEEK': [week],
                    'SCHEDULED_DEPARTURE': [scheduled_departure_minutes],
                    'DEPARTURE_TIME': [departure_time_minutes],
                    'WHEELS_OFF': [wheels_off_minutes],
                    'TAXI_OUT': [taxi_out_time],
                    'TAXI_IN': [taxi_in_time],
                    'DEPARTURE_DELAY': [departure_delay]
                })

                # Encode categorical features
                for col in columns_to_encode:
                    if col in input_data.columns:
                        encoder = label_encoders[col]
                        input_data[col] = encoder.transform(input_data[col])

                # Ensure all required columns are present and in the correct order
                input_data = input_data.reindex(columns=all_columns, fill_value=0)

                # Scale the data
                input_data_scaled = scaler.transform(input_data)
                input_data_scaled = pd.DataFrame(input_data_scaled, columns=all_columns)

                # Make predictions
                logistic_predictions = logistic_model.predict(input_data_scaled)
                logistic_predictions = np.where(logistic_predictions == 1, 'Delayed', 'Not Delayed')
                xgboost_predictions = xgboost_model.predict(input_data_scaled)

                # Prepare results
                results = pd.DataFrame({
                    'IS_DELAYED': logistic_predictions,
                    'ARRIVAL_DELAY': xgboost_predictions,
                    'AIRLINE': [airline],
                    'ORIGIN_AIRPORT': [origin_airport],
                    'DESTINATION_AIRPORT': [destination_airport],
                    'WEEK': [week],
                    'SCHEDULED_DEPARTURE': [scheduled_departure_minutes],
                    'DEPARTURE_TIME': [departure_time_minutes],
                    'WHEELS_OFF': [wheels_off_minutes],
                    'TAXI_OUT': [taxi_out_time],
                    'TAXI_IN': [taxi_in_time],
                    'DEPARTURE_DELAY': [departure_delay],
                    'DATE': [date]
                })

                st.write("Prediction Results:")
                # Apply color styling to the 'IS_DELAYED' and 'ARRIVAL_DELAY' columns 
                styled_results = results.style.applymap(lambda v: 'color: red' if v == 'Delayed' else 'color: white' if isinstance(v, str) else '', subset=['IS_DELAYED']).applymap(lambda v: 'color: red' if v > 0 else 'color: white' if isinstance(v, (int, float)) else '', subset=['ARRIVAL_DELAY'])
                st.dataframe(styled_results)

            except Exception as e:
                st.error(f"Error in making prediction: {str(e)}")

if __name__ == '__main__':
    main()
