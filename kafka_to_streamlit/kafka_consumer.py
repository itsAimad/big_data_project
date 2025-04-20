from kafka import KafkaConsumer
import json

class Consumer:
    def __init__(self):
        self.consumer = None  # Initialize a consumer attribute

    def kafka_initialization(self, topic, bootstrap_servers):
        try:
            # Initialize KafkaConsumer
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest', 
                enable_auto_commit=True     
            )
            return self.consumer
        except Exception as e:
            print(f"Error initializing KafkaConsumer: {e}")
            return None

    def get_data(self):
        try:
            if not self.consumer:
                raise RuntimeError("Consumer not initialized. Call kafka_initialization() first.")

            # Fetch messages from the KafkaConsumer
            for message in self.consumer:
                return message.value  # Return the value of the first message
            return None
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None