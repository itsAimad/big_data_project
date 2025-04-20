import streamlit as st
from kafka import KafkaConsumer
import json
import pandas as pd
import base64
import os

custom_css = """
    <style>
        #breast-cancer-image{
 
            border-radius: 10px;
            margin-top:20px;
          
        }
    </style>

"""


def get_data_from_kafka():
    consumer = KafkaConsumer(
        'prediction-results-topic',
        bootstrap_servers='hadoop-master:9092',
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        consumer_timeout_ms=1000  # Prevent blocking indefinitely
    )
    return consumer

sick = 0
healthy = 0


def main():

    st.set_page_config(page_icon="🏥", page_title="Real-time Breast Cancer App 🩺",
                       layout='wide',
                       initial_sidebar_state='expanded')
    st.title("Real-time Breast Cancer Predictions")

    # Render Css
    st.markdown(custom_css, unsafe_allow_html=True)

    tab_titles = ["Overview","Data Visualization", "Real-Time Predictions","Data Insights","Test a Case"]

    tab1,tab2,tab3,tab4,tab5 = st.tabs(tab_titles)
    with tab1:
        # Divide the page into two columns: description (left) and image (right)
        col1,col2 = st.columns([3,1])
        with col1:
            st.markdown("""
            ### Welcome to the Real-time Breast Cancer Predictor App 🩺
            This application leverages machine learning and real-time data streaming 
            to assist healthcare professionals in diagnosing breast cancer effectively.
            
            - **Streamlit**: A fast and interactive web-based tool.
            - **Logistic Regression**: A machine learning algorithm for accurate predictions.
            - **Kafka Streaming**: Enables real-time data processing.
            
            By integrating advanced technologies like logistic regression, data visualization, and Kafka streaming, this app empowers early detection and improves patient outcomes.
            """)
        image_path = '/root/myproject/kafka_to_streamlit/images/image1.png'
        try:
            with open(image_path,"rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                col2.markdown(f"""<img src="data:image/png;base64,{encoded_image}" alt="Breast Cancer Image" id="breast-cancer-image">""",unsafe_allow_html=True)
        except FileNotFoundError:
            col2.error(f"Image not found at : {os.path.abspath(image_path)}")
            return
    
    
    
    
    consumer = get_data_from_kafka()
    try:
        
        for message in consumer:
            data = message.value

            st.dataframe(data)

        # st.rerun()
    except Exception as e:
        st.error(e)

if __name__ == "__main__":
    main()