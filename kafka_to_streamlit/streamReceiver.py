import streamlit as st
from kafka import KafkaConsumer
import json


consumer = KafkaConsumer("aimad",bootstrap_servers="hadoop-master:9092",value_deserializer=lambda v: json.loads(v.decode('utf-8')))

st.header("KAFKA REAL TIME RECEIVING DATA ✔️")
for message in consumer:

        st.text(f"New Message : {message.value}")

