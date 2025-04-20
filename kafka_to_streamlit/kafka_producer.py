import random
from kafka import KafkaProducer
import json
from time import sleep

class ProduceData():
    def __init__(self, bootstrap_servers, topic):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic

    def generate_ambiguous_sample(self):
        """Generate features that are more ambiguous between benign and malignant"""
        # Using ranges that overlap between benign and malignant cases
        return {
            "radius_mean": round(random.uniform(12.0, 15.0), 4),  
            "texture_mean": round(random.uniform(15.0, 18.0), 4), 
            "perimeter_mean": round(random.uniform(80.0, 95.0), 4), 
            "area_mean": round(random.uniform(450.0, 650.0), 4), 
            "smoothness_mean": round(random.uniform(0.085, 0.105), 4),
            "compactness_mean": round(random.uniform(0.070, 0.100), 4),
            "concavity_mean": round(random.uniform(0.050, 0.100), 4),  
            "concave_points_mean": round(random.uniform(0.020, 0.060), 4),
            "symmetry_mean": round(random.uniform(0.150, 0.180), 4),  
            "fractal_dimension_mean": round(random.uniform(0.055, 0.065), 4),
            "radius_se": round(random.uniform(0.600, 0.900), 4),  
            "texture_se": round(random.uniform(1.000, 1.500), 4), 
            "perimeter_se": round(random.uniform(4.000, 6.000), 4),
            "area_se": round(random.uniform(35.0, 55.0), 4),  
            "smoothness_se": round(random.uniform(0.008, 0.015), 4),
            "compactness_se": round(random.uniform(0.015, 0.030), 4),
            "concavity_se": round(random.uniform(0.080, 0.150), 4),  
            "concave_points_se": round(random.uniform(0.008, 0.015), 4),
            "symmetry_se": round(random.uniform(0.015, 0.025), 4),  
            "fractal_dimension_se": round(random.uniform(0.002, 0.004), 4),
            "radius_worst": round(random.uniform(13.0, 16.0), 4),  
            "texture_worst": round(random.uniform(18.0, 22.0), 4),  
            "perimeter_worst": round(random.uniform(90.0, 105.0), 4), 
            "area_worst": round(random.uniform(600.0, 800.0), 4), 
            "smoothness_worst": round(random.uniform(0.100, 0.130), 4),
            "compactness_worst": round(random.uniform(0.150, 0.250), 4),
            "concavity_worst": round(random.uniform(0.150, 0.250), 4),
            "concave_points_worst": round(random.uniform(0.050, 0.100), 4), 
            "symmetry_worst": round(random.uniform(0.200, 0.250), 4),  
            "fractal_dimension_worst": round(random.uniform(0.065, 0.075), 4)  
        }

    def kafka_produce(self):
        producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        x = 1
        while True:
            # Generate ambiguous data
            data = self.generate_ambiguous_sample()
            producer.send(self.topic, value=data)
            print(f"Sent sample N°{x} 😊 - Ambiguous Case")
            sleep(2)
            x += 1

if __name__ ==  "__main__":
    producer = ProduceData(bootstrap_servers="hadoop-master:9092",topic="prediction-topic")
    producer.kafka_produce()