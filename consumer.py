from kafka import KafkaConsumer
import json

consumer = KafkaConsumer("aimad",bootstrap_servers="hadoop-master:9092",value_deserializer = lambda v: json.loads(v.decode("utf-8")))


while True:
    for message in consumer:
        print(f"New Message ✔️ : {message.value}")