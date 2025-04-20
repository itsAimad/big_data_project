from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(bootstrap_servers="hadoop-master:9092",value_serializer=lambda v: json.dumps(v).encode('utf-8'))

while True:
    message= input(" > Enter a message : ")

    topic = "aimad"
    producer.send("aimad",value=message)
   