import streamlit as st
import json
import pandas as pd
import base64
import os
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField,DoubleType,IntegerType
from pyspark.ml.feature import MinMaxScalerModel,VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
import plotly.graph_objects as go
import time
from kafka_consumer import Consumer
from datetime import datetime

# Define the numerical columns used in the model
numerical_cols = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se",
    "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

## Initialize SparkSession
spark = SparkSession.builder \
            .appName("Breast Cancer Classification") \
            .master("local[*]") \
            .getOrCreate()
### Load Saved Model, MinMaxScaler
MODEL = LogisticRegressionModel.load("file:///root/myproject/Model/LR")

SCALER_MODEL = MinMaxScalerModel.load("file:///root/myproject/Model/Scaler")
custom_css = """
    <style>
        #breast-cancer-image{
 
            border-radius: 10px;
            margin-top:20px;
            height:400px;
            width:390px;
            transition: transform 0.6s ease-out, filter 0.6s ease-out;
            
            
            
        }
        #breast-cancer-image:hover{
            transform: scale(1.05);
            cursor:pointer;
            filter: drop-shadow(0px 0px 20px #164970);
        }

        @media(max-width:800px){
            #breast-cancer-image{
                height:100%;
                width:100%;
            }
        }

         .techUsed {
            overflow: hidden;
            width: 100%; 
            position: relative;
            padding: 10px 0;
        }
        
        .techUsed h6 {
            margin-top:10px;
        }
        
        .techUsed .images {
            display: flex;
            animation: scroll 20s linear infinite;
            width: max-content; 
        }

        .techUsed .images img {
            margin-right: 20px; 
        }
        .techUsed .images img:hover{
            transform: translateY(-4px);
            filter:drop-shadow(0px 0px 12px #fff);
            transition: transform 0.6s ease-out, filter 0.6s ease-out;
        }
        @keyframes scroll {
            0% {
                transform: translateX(0);
            }
            100% {
                transform: translateX(-50%); 
            }
        }

        .container2{
            background: linear-gradient(to left,red,white);
            padding:15px 20px;
            border-radius: 18px;
            width:290px;
            transition: transform 0.3s ease-in-out,filter 0.3s ease-in;
           
            }
        .container2 h2{
            text-align:center;
            color:#000;
          
        }
        .container2 p{
                color:#000;
                font-weight:600;
                
            }

        .container2:hover{
            transform: scale(1.05);
            filter: drop-shadow(0px 0px 14px #fff);
           
        }
        #benign{
            color: rgb(0,210,0);
            padding: 3px 9px;
            background-color: #000;
            border-radius: 9px;
        }
        #malicious{
            color: rgb(210,0,0);
            padding: 3px 9px;
            background-color: #000;
            border-radius: 9px;
        }

        #resultM{
            background-color: rgb(210,0,0);
            padding:3px 6px;
            color:#fff;
            font-weight:600;
            width:120px;
            border-radius:19px;
            text-align:center;
            margin-left:64px;
        }

        #resultB{
                background-color: rgb(0,210,0);
            padding:3px 6px;
            color:#fff;
            font-weight:600;
            width:120px;
            border-radius:19px;
            text-align:center;
            margin-left:64px;
            }
    </style>

"""


## load dataset
def load_dataset():
    df = pd.read_csv("file:///root/myproject/Data/data.csv")
    df = df.drop(["Unnamed: 32","id"],axis=1)
    
    df = df.rename(columns={
        "concave points_mean" : "concave_points_mean",
        "concave points_se" : "concave_points_se",
        "concave points_worst" : "concave_points_worst"
    })
    
    return df
### Generate the graphs in Data Visualization
def feature_distribution_plot(feature):
    data = load_dataset()
    fig = px.histogram(
        data, x=feature, color='diagnosis',
        marginal='box',
        color_discrete_map={'B': '#00CC66', 'M': '#FF0000'},
        labels={'diagnosis': 'Diagnosis'},
        barmode='overlay',
        opacity=0.7,
        title=f"Distribution of {feature} by Diagnosis"
    )
    
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title="Count",
        legend_title="Diagnosis",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemsizing='constant'
        )
    )
    
    # Update legend labels
    if len(fig.data) > 0:
        if fig.data[0].name =='B':
            fig.data[0].name = 'Benign'
    if len(fig.data) > 1:
        if fig.data[1].name == 'M':
            fig.data[1].name='Malignant'
    
    return fig


############3
## Correlation heatmap of mean features
def plotly_heatmap():
    df = load_dataset()

    mean_cols = [col for col in df.columns if 'mean' in col]

    corr = df[mean_cols].corr()

    fig = px.imshow(corr,
                    title="Correlation between Mean Features",
                    color_continuous_scale='RdBu_r',
                    zmin=-1,zmax=1
                    )
    fig.update_layout(height=600)
    
    return fig

###################333#########

## Create 3d  scatter visualization
def create_3d_scatters():
    data = load_dataset()

    fig = px.scatter_3d(
        data, 
        x='radius_mean', 
        y='concave_points_mean', 
        z='area_mean',
        color='diagnosis',
        # color_continuous_scale=['#00CD66', '#FF0000'],
        color_discrete_map= {"A": "#EE0A0A","B" : "green"},
        opacity=0.7,
        title="3D Visualization of Key Features"
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Radius Mean',
            yaxis_title='Concave Points Mean',
            zaxis_title='Area Mean'
        ),
        coloraxis_colorbar=dict(
            title="Diagnosis",
            tickvals=[0, 1],
            ticktext=["Benign", "Malignant"],
        )
    )

    
    return fig




def generate_scatters(col1,col2,color):
    df = load_dataset()

    fig = px.scatter(df,
                     x=col1,
                     y=col2,
                     color=color,
                    color_discrete_map={"A": "#FFCC00", "B": "#2F71B6", "C": "green"},
                    opacity=0.7)

    
    return fig


def generate_Bar(col1,col2,color,barmode):
    df = load_dataset()
    fig = px.bar(df,x=col1,
                 y=col2,
                 color=color,
                 barmode=barmode,
                 color_discrete_map={"A":"#2F71B6","B":"#6B0BB5"})
    
    return fig

def generate_hist(col1,col2,color,marginal):
    df = load_dataset()
    fig = px.histogram(df,
                  x=col1,
                  y=col2,
                  color=color,
                  color_discrete_map={"A":"#2F71B6", "B": "#6B0BB5"},
                  marginal=marginal,
                  hover_data=df.columns)
    return fig

def generate_box(col1,col2,color):
    df = load_dataset()
    fig = px.box(df,
                 x=col1,
                 y=col2,
                 color=color,
                 color_discrete_map={"A" : "#2F71B6", "B" : "#6B0BB5"})
    
    return fig

def generate_heatmap(col1,col2):
    df = load_dataset()
    fig = px.density_heatmap(
        df,
        x=col1,
        y=col2,
        marginal_x="rug",
        marginal_y="histogram",color_continuous_scale=["#2F71B6","#6B0BB5"]
    )
    return fig

###################################
### Adding sidebar with sliders of each feature, for test a case tab
def add_sidebar():
    st.sidebar.header("By: AIMAD BOUYA")
    st.sidebar.header("Cell Nuclei Measurements")

    data = load_dataset()

    input_dict = {}

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave_points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)","texture_se"),
        ("Perimeter (se)","perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave_points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)","area_worst"),
        ("Smothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave_points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label=label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
            )
      
    return input_dict



def get_scaled_values(input_dict):
    
    schema = StructType([StructField(key, DoubleType(), True) for key in input_dict.keys()])
    input_df = spark.createDataFrame([input_dict], schema=schema)
    
    
    feature_columns = input_df.columns
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_df = assembler.transform(input_df)
    
    scaled_df = SCALER_MODEL.transform(assembled_df)
    

    scaled_features = scaled_df.select("scaled_features").collect()[0][0].toArray()
    
    scaled_dict = dict(zip(input_dict.keys(), scaled_features))
    
    return scaled_dict

    # Initialize SparkSession
    # spark = SparkSession.builder.appName("StreamlitApp").getOrCreate()

    # # Debug: Print input dictionary
    # # print("Input Dictionary:", input_dict)

    # # Define schema
    # schema = StructType([StructField(key, DoubleType(), True) for key in input_dict.keys()])
    # # print("Schema:", schema)

    # # Create input DataFrame
    # input_df = spark.createDataFrame([input_dict], schema=schema)
    # # print("Input DataFrame:")
    # # input_df.show()

    # # Use VectorAssembler to create a "features" column
    # feature_columns = input_df.columns
    # assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    # assembled_df = assembler.transform(input_df)
    # # print("Assembled DataFrame:")
    # # assembled_df.show()

    # # Apply MinMaxScalerModel
    # scaled_df = SCALER_MODEL.transform(assembled_df)
    # # print("Scaled DataFrame:")
    # # scaled_df.show()

    # # Debug: Print the scaled features
    # scaled_features = scaled_df.select("scaled_features").collect()[0][0]
    # # print("Scaled Features:", scaled_features)

    # # Collect scaled values
    # scaled_values = scaled_df.collect()[0].asDict()
    # # print("Scaled Values:", scaled_values)

    # return scaled_values
##############################
#### TAB 5
def radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Area','Perimeter','Texture',
                  'Radius','Fractal Dimension',
                   'Symmetry','Concave Points','Concavity',
                   'Compactness','Smoothness']
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['area_mean'],input_data['perimeter_mean'],
            input_data['texture_mean'],input_data['radius_mean'],
            input_data['fractal_dimension_mean'], input_data['symmetry_mean'],
            input_data['concave_points_mean'],input_data['concavity_mean'],
            input_data['compactness_mean'],input_data['smoothness_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['area_se'],input_data['perimeter_se'],
            input_data['texture_se'],input_data['radius_se'],
            input_data['fractal_dimension_se'], input_data['symmetry_se'],
            input_data['concave_points_se'],input_data['concavity_se'],
            input_data['compactness_se'],input_data['smoothness_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r = [
            input_data['area_worst'],input_data['perimeter_worst'],
            input_data['texture_worst'],input_data['radius_worst'],
            input_data['fractal_dimension_worst'], input_data['symmetry_worst'],
            input_data['concave_points_worst'],input_data['concavity_worst'],
            input_data['compactness_worst'],input_data['smoothness_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar = dict(
            radialaxis=dict(
                visible=True,
                range=[0,1]
            )),
            showlegend=True,
            
    )

    return fig
###########################33
### Prediction container
def add_predictions(input_data, spark):
    
    schema = StructType([StructField(key, DoubleType(), True) for key in input_data.keys()])

    input_df = spark.createDataFrame([input_data],schema=schema)

    # Use VectorAssembler to create a feature column
    feature_columns = input_df.columns 
    assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")
    assembled_df = assembler.transform(input_df)

    scaled_df = SCALER_MODEL.transform(assembled_df)
    # Make Predictions using the loaded logistic REgression models
    predictions = MODEL.transform(scaled_df)
    
    prediction = predictions.select("prediction").collect()[0][0]
    probability = predictions.select("probability").collect()[0][0]

    if prediction == 0:
        st.markdown(f"""<div class='container2'>
                    <h2>Cell Cluster predictions</h2>
                    <p>The result is : </p>
                    <p id='resultB'>Benign</p>
                    <p>Probability of being benign : <span id='benign'>{probability[0]}</span></p>
                    <p>Probability of being Malicous : <span id='malicious'>{probability[1]}</span></p>
                    <p>This app can assist medical professionals in making a diagnosis, but should not be used as a substitude for a professional diagnosis.</p>
                    </div>
                    """,unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='container2'>
                    <h2> Cell Cluster predictions </h2>
                    <p>The result is : </p>
                    <p id='resultM'>Malicious</p>
                    <p>Probability of being benign : <span id='benign'>{probability[0]} </span> </p>
                    <p>Probability of being Malicious : <span id='malicious'>{probability[1]} </span> </p>
                    <p>This app can assist medical professionals in making a diagnosis, but should not be used as a substitude for a professional diagnosis.</p>
                    </div>
                    """,unsafe_allow_html=True)

def check_kafka_connection():
    """Check if Kafka connection is available"""
    try:
        consumer = Consumer()
        consumer.kafka_initialization('prediction-results-topic', 'hadoop-master:9092')
        consumer.consumer.close()
        return True
    except Exception:
        return False
    
def get_prediction_with_timeout(timeout_seconds):
    """Get prediction data from Kafka with timeout"""
    try:
        consumer = Consumer()
        consumer.kafka_initialization('prediction-results-topic', 'hadoop-master:9092')
        
        start_time = time.time()
        message = None
        
        while time.time() - start_time < timeout_seconds:
            message = consumer.get_data()
            if message:
                consumer.consumer.close()
                try:
                    # Parse the message
                    if isinstance(message, str):
                        data = json.loads(message)
                    else:
                        data = message
                        
                    # Extract probability vector from the prediction_probability field
                    prob_vector = data.get('prediction_probability', {})
                    if isinstance(prob_vector, str):
                        prob_vector = json.loads(prob_vector)
                    
                    # Ensure probability is in the correct format
                    if isinstance(prob_vector, dict) and 'values' in prob_vector:
                        probabilities = prob_vector['values']
                    elif isinstance(prob_vector, list):
                        probabilities = prob_vector
                    else:
                        probabilities = [0.0, 0.0]
                    
                    # Update the data with properly formatted probabilities
                    data['prediction_probability'] = probabilities
                    return data
                    
                except json.JSONDecodeError as e:
                    st.error(f"Error parsing message: {str(e)}")
                    return None
            time.sleep(0.1)
        
        consumer.consumer.close()
        return None
        
    except Exception as e:
        st.error(f"Error in Kafka consumer: {str(e)}")
        return None

def display_prediction(prediction_data):
    if prediction_data:
        # Get the prediction label
        predicted_label = prediction_data.get('predicted_label', 0)
        prediction = "Malignant" if predicted_label == 1 else "Benign"
        
        # Get probabilities
        probabilities = prediction_data.get('prediction_probability', [0.0, 0.0])
        if isinstance(probabilities, (list, tuple)) and len(probabilities) == 2:
            confidence = probabilities[1] if prediction == "Malignant" else probabilities[0]
            confidence = float(confidence) * 100  # Convert to percentage
        else:
            confidence = 0.0
            
        st.markdown(f"""
            <div style="background: {'#e74c3c' if prediction == 'Malignant' else '#2ecc71'}; 
                padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
                <h3 style="color: white; margin: 0;">
                    Prediction: {prediction}
                </h3>
                <p style="font-size: 1.2em; margin: 10px 0; color: white">
                    Confidence: {confidence:.1f}%
                </p>
            </div>
        """, unsafe_allow_html=True)

def main():
    consumer = Consumer()
    try:
        consumer.kafka_initialization('prediction-results-topic', 'hadoop-master:9092')
        st.session_state.kafka_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize Kafka consumer: {str(e)}")
        st.session_state.kafka_initialized = False
    st.set_page_config(page_icon="🏥", page_title="Real-time Breast Cancer App 🩺",
                       layout='wide',
                       initial_sidebar_state='expanded')
    st.title("🏥 Real-time Breast Cancer Monitoring System")

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
            
            
            By integrating advanced technologies like logistic regression, data visualization, and Kafka streaming, this app empowers early detection and improves patient outcomes.
            """)

            st.markdown("<h3>System Architecture </h3>",unsafe_allow_html=True)
            st.image("images/bigDataArchitecture.png")

        with col2:
            image_path = '/root/myproject/kafka_to_streamlit/images/image1.png'
            try:
                with open(image_path,"rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    st.markdown(f"""<img src="data:image/png;base64,{encoded_image}" alt="Breast Cancer Image" id="breast-cancer-image">""",unsafe_allow_html=True)
            except FileNotFoundError:
                col2.error(f"Image not found at : {os.path.abspath(image_path)}")
                return
            
            st.markdown(f"""         
                 <div class="techUsed">
                    <h6>Technologies Used ⚙️</h6>
                <div class="images">
                    <img src="https://www.vectorlogo.zone/logos/python/python-icon.svg" width="36" height="36">
                    <img src="https://www.vectorlogo.zone/logos/apache_spark/apache_spark-ar21.svg" height="60" width="60" style="position:relative;bottom:10px;">
                    <img src="https://www.vectorlogo.zone/logos/apache_kafka/apache_kafka-ar21.svg" width="53" height="40">
                    <img src="https://www.vectorlogo.zone/logos/plotly/plotly-ar21.svg" width="53" height="40">
                    <img src="https://www.vectorlogo.zone/logos/grafana/grafana-icon.svg" width="36" height="36">
                    <img src="https://www.vectorlogo.zone/logos/docker/docker-icon.svg" width="45" height="45">
                    <img src="https://www.vectorlogo.zone/logos/postgresql/postgresql-icon.svg" width='40' height='40'>
                    <img src="https://www.vectorlogo.zone/logos/python/python-icon.svg" width="36" height="36"> <!--e-->
                    <img src="https://www.vectorlogo.zone/logos/apache_spark/apache_spark-ar21.svg" height="60" width="60" style="position:relative;bottom:10px;">
                    <img src="https://www.vectorlogo.zone/logos/apache_kafka/apache_kafka-ar21.svg" width="53" height="40">
                    <img src="https://www.vectorlogo.zone/logos/plotly/plotly-ar21.svg" width="53" height="40">
                    <img src="https://www.vectorlogo.zone/logos/grafana/grafana-icon.svg" width="36" height="36">
                    <img src="https://www.vectorlogo.zone/logos/docker/docker-icon.svg" width="45" height="45">
                    <img src="https://www.vectorlogo.zone/logos/postgresql/postgresql-icon.svg" width='40' height='40'>
                </div>
    </div>
            """,unsafe_allow_html=True)
    

    with tab2:
        st.markdown("<h3>Data Visualization</h3>",unsafe_allow_html=True)
        col1, col2 = st.columns([2,2])

        with col1:
           
            st.plotly_chart(feature_distribution_plot('radius_mean'),use_container_width=True)

        with col2:
           
            st.plotly_chart(feature_distribution_plot('texture_mean'),use_container_width=True)

        # Create Correlation HeatMap between mean features
        st.plotly_chart(plotly_heatmap())
    
        st.plotly_chart(create_3d_scatters())

        with st.container():
            colL,colR = st.columns([1,2])

            with colL:
                df = load_dataset()
                st.write("You Can Choose the graph 📊")
                option = st.selectbox("Make your Choice 😊",
                                      options=["Scatters","Histogram",
                                      "Bar","BoxPlot","density_heatmap",
                                      "Line"])
                if option == "Scatters":
                    col1 = st.selectbox("First Column",options=df.columns)
                    col2 = st.selectbox("Second Column", options=df.columns)
                    color = st.selectbox("Color", options=df.columns)
                elif option == "Bar":
                    col1 = st.selectbox("First Column",options=df.columns)
                    col2 = st.selectbox("Second Column",options=df.columns)
                    color = st.selectbox("Color",options=df.columns)
                    barmode = st.selectbox("BarMode",options=["group","stack","overlay","relative"])

                elif option == "Histogram":
                    col1 = st.selectbox("First Column",options=df.columns)
                    col2 = st.selectbox("Second Column", options=df.columns)
                    color = st.selectbox("Color", options=df.columns)
                    marginal = st.selectbox("Marginal",options=["rug","box"])
                
                elif option == "BoxPlot":
                    col1 = st.selectbox("First Column",options=df.columns)
                    col2 = st.selectbox("Second Column", options=df.columns)
                    color = st.selectbox("Color", options=df.columns)

                elif option == "density_heatmap":
                    col1 = st.selectbox("First Column",options=df.columns)
                    col2 = st.selectbox("Second Column", options=df.columns)
                elif option == 'Line':
                    col1 = st.selectbox("First Column",options=df.columns)
                    col2 = st.selectbox("Second Column", options=df.columns)
                submit = st.button("Generate ✔️")
                
            with colR:
                if submit:
                    if option == "Scatters":
                        st.plotly_chart(generate_scatters(col1,col2,color))
                    elif option == "Bar":
                        st.plotly_chart(generate_Bar(col1,col2,color,barmode))
                    elif option == "Histogram":
                        st.plotly_chart(generate_hist(col1,col2,color,marginal))
                    elif option == "BoxPlot":
                        st.plotly_chart(generate_box(col1,col2,color))
                    elif option == "density_heatmap":
                        st.plotly_chart(generate_heatmap(col1,col2))
                    elif option == 'Line':
                        st.plotly_chart()

      
    def save_to_csv(message_data, csv_path='/root/myproject/kafka_to_streamlit/predictions_history.csv'):
        """Save prediction data to CSV file with proper data handling"""
        try:
            # Parse the message data
            if isinstance(message_data, str):
                prediction_data = json.loads(message_data)
            else:
                prediction_data = message_data

            # Get probabilities from the prediction_probability field
            probabilities = prediction_data.get('prediction_probability', [0.0, 0.0])
            if isinstance(probabilities, dict) and 'values' in probabilities:
                probabilities = probabilities['values']
            
            # Create base dictionary with prediction info
            new_row = {
                'timestamp': datetime.now(),
                'predicted_label': float(prediction_data.get('predicted_label', 0)),
                'benign_probability': float(probabilities[0]),
                'malignant_probability': float(probabilities[1])
            }

            # Add feature columns
            for col in numerical_cols:  # numerical_cols should be defined globally
                new_row[col] = float(prediction_data.get(col, 0.0))

            # Create DataFrame with the new row
            new_df = pd.DataFrame([new_row])

            # Load existing data or create new DataFrame
            if os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path)
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                updated_df = new_df

            # Save to CSV
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            updated_df.to_csv(csv_path, index=False)
            return updated_df

        except Exception as e:
            st.error(f"Error processing message data: {str(e)}")
            return None
    
    def create_prediction_distribution_chart(df):
        """Create a pie chart showing distribution of predictions"""
        prediction_counts = df['predicted_label'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['Benign', 'Malignant'],
            values=[
                prediction_counts.get(0, 0),
                prediction_counts.get(1, 0)
            ],
            hole=.3,
            marker_colors=['#2ecc71', '#e74c3c']
        )])
        fig.update_layout(
            title="Distribution of Predictions",
            showlegend=True,
            height=300
        )
        return fig
  
    def create_prediction_evolution_chart(df):
        """Create a line plot showing the evolution of predictions over time with dark mode"""
        # Convert timestamp to datetime if it's not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create separate counts for each class
        benign_counts = df[df['predicted_label'] == 0].groupby('timestamp').size().cumsum()
        malignant_counts = df[df['predicted_label'] == 1].groupby('timestamp').size().cumsum()
        
        # Create the line plot
        fig = go.Figure()
        
        # Add traces for both classes
        fig.add_trace(go.Scatter(
            x=benign_counts.index,
            y=benign_counts.values,
            name='Benign',
            line=dict(color='#2ecc71', width=2),
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=malignant_counts.index,
            y=malignant_counts.values,
            name='Malignant',
            line=dict(color='#e74c3c', width=2),
            mode='lines+markers'
        ))
        
        # Update layout for dark mode
        fig.update_layout(
            title=dict(
                text='Cumulative Cases Over Time',
                font=dict(color='white')  # Set title text color to white
            ),
            xaxis_title=dict(
                text='Time',
                font=dict(color='white')  # Set x-axis text color to white
            ),
            yaxis_title=dict(
                text='Number of Cases',
                font=dict(color='white')  # Set y-axis text color to white
            ),
            hovermode='x unified',
            height=400,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(color='white')  # Set legend text color to white
            ),
            
            plot_bgcolor='#333333',  # Set plot background to dark gray
            paper_bgcolor='#1e1e1e',  # Set paper background to even darker gray
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#4d4d4d',  # Set gridlines to lighter gray
                color='white'  # Set x-axis tick labels to white
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#4d4d4d',  # Set gridlines to lighter gray
                color='white'  # Set y-axis tick labels to white
            )
        )
        
        return fig
    

    def create_timeline_chart(df):
        """Create a timeline of predictions"""
        fig = px.line(
            df,
            x='timestamp',
            y=['benign_probability', 'malignant_probability'],
            title='Prediction Probabilities Over Time',
            labels={'value': 'Probability', 'timestamp': 'Time'},
            height=300
        )
        fig.update_layout(hovermode='x unified')
        return fig

    def create_feature_histogram(df, feature_name):
        """Create histogram for a specific feature with diagnosis coloring"""
        fig = px.histogram(
            df,
            x=feature_name,
            color='predicted_label',
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
            labels={
                'predicted_label': 'Diagnosis',
                'count': 'Count',
                feature_name: feature_name.replace('_', ' ').title()
            },
            title=f'Distribution of {feature_name.replace("_", " ").title()}',
            height=400,
            marginal='box'  # Add box plot on the marginal
        )
        
        # Update layout for better visualization
        fig.update_layout(
            bargap=0.1,
            bargroupgap=0.1,
            legend=dict(
                title="Diagnosis",
                itemsizing='constant',
                itemwidth=30,
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def handle_real_time_monitoring():
        """Handle real-time monitoring logic with proper state management"""
        # Initialize session state if not exists
        if 'real_time_active' not in st.session_state:
            st.session_state.real_time_active = False
        
        # Toggle for real-time mode
        real_time_active = st.toggle(
            '🔄 Activate Real-time Predictions',
            value=st.session_state.real_time_active,
            help="Toggle to start receiving predictions in real-time",
            key='realtime_toggle'
        )
        
        # Update session state
        st.session_state.real_time_active = real_time_active
        
        if real_time_active:
            try:
                status_placeholder = st.empty()
                status_placeholder.info("🔄 Monitoring for new predictions...")
                
                # Get current row count
                current_count = len(st.session_state.predictions_df) if not st.session_state.predictions_df.empty else 0
                
                # Get new prediction with timeout
                message = get_prediction_with_timeout(timeout_seconds=1)
                if message:
                    updated_df = save_to_csv(message)
                    if updated_df is not None:
                        # Check if new data was actually added
                        new_count = len(updated_df)
                        if new_count > current_count:
                            st.session_state.predictions_df = updated_df
                            st.session_state.last_update = datetime.now()
                            st.cache_data.clear()
                            status_placeholder.success("✨ New prediction received!")
                            time.sleep(0.5)  # Small delay to show the success message
                            st.rerun()
                
                # Add a small delay to prevent excessive updates
                time.sleep(1)
                st.rerun()
            
            except Exception as e:
                st.error(f"Error in monitoring: {str(e)}")
                time.sleep(1)
                st.rerun()
        else:  # Manual mode
            if st.button("🔄 Manual Refresh", key='manual_refresh'):
                message = get_prediction_with_timeout(timeout_seconds=1)
                if message:
                    updated_df = save_to_csv(message)
                    if updated_df is not None:
                        st.session_state.predictions_df = updated_df
                        st.session_state.last_update = datetime.now()
                        st.cache_data.clear()
                        st.success("✨ New prediction received!")
                        st.rerun()
                else:
                    st.warning("⚠️ No new predictions available")
                

    with tab3:
        st.markdown("<h3 style='color: #1E88E5;'>Real-Time Predictions Dashboard</h3>", unsafe_allow_html=True)
        
        # Initialize session state variables
        if 'predictions_df' not in st.session_state:
            try:
                if os.path.exists('/root/myproject/kafka_to_streamlit/predictions_history.csv'):
                    st.session_state.predictions_df = pd.read_csv('/root/myproject/kafka_to_streamlit/predictions_history.csv')
                    st.session_state.predictions_df['timestamp'] = pd.to_datetime(st.session_state.predictions_df['timestamp'])
                else:
                    # Create empty DataFrame with all required columns
                    columns = ['timestamp', 'predicted_label', 'benign_probability', 'malignant_probability']
                    features = [
                        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                        'smoothness_mean', 'compactness_mean', 'concavity_mean',
                        'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                        'smoothness_se', 'compactness_se', 'concavity_se',
                        'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
                        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                        'smoothness_worst', 'compactness_worst', 'concavity_worst',
                        'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
                    ]
                    columns.extend(features)
                    st.session_state.predictions_df = pd.DataFrame(columns=columns)
            except Exception as e:
                st.error(f"Error initializing predictions DataFrame: {str(e)}")
                st.session_state.predictions_df = pd.DataFrame()

        if 'real_time_active' not in st.session_state:
            st.session_state.real_time_active = False

        # Dashboard Layout
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("""
                <div style="background: #f0f7ff; padding: 15px; border-radius: 10px; border-left: 5px solid #1E88E5;">
                    <h4 style='margin: 0; color: #1E88E5;'>Control Panel</h4>
                </div>
            """, unsafe_allow_html=True)
            
            # Real-time monitoring toggle
            handle_real_time_monitoring()

            # Summary Statistics
            if not st.session_state.predictions_df.empty:
                st.markdown("""
                    <div style="background: #e8f5e9; padding: 15px; border-radius: 10px; margin-top: 20px;">
                        <h4 style='margin: 0; color: #2e7d32;'>Summary Statistics</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                total_predictions = len(st.session_state.predictions_df)
                benign_count = (st.session_state.predictions_df['predicted_label'] == 0).sum()
                malignant_count = total_predictions - benign_count
                
                st.metric("Total Cases", total_predictions)
                st.metric("Active Session", f"{datetime.now().strftime('%H:%M:%S')}")
                st.metric("Benign Cases", benign_count, 
                        delta=f"{(benign_count/total_predictions*100):.1f}%",
                        delta_color="normal")
                st.metric("Malignant Cases", malignant_count,
                        delta=f"{(malignant_count/total_predictions*100):.1f}%",
                        delta_color="inverse")
                
                coll,colr = st.columns([1,1])
                with coll:
                    if st.button("📥 Export Data"):
                        csv = st.session_state.predictions_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="predictions_export.csv",
                            mime="text/csv"
                        )
                with colr:
                    if st.button("📈 Export Graph (HTML)"):
                        evolution_fig = create_prediction_evolution_chart(st.session_state.predictions_df)
                        evolution_fig.update_layout(
                            paper_bgcolor='white',
                            plot_bgcolor='white',
                            title=dict(font=dict(color='black')),
                            xaxis=dict(color='black',gridcolor='lightgray'),
                            yaxis=dict(color='black',gridcolor='lightgray'),
                            legend=dict(font=dict(color='black'))
                        )
                        html = evolution_fig.to_html()  # Fixed: use to_html() instead of add_annotation
                        st.download_button(
                            label='Download HTML',
                            data=html,
                            file_name='evolution_graph.html',
                            mime='text/html'
                        )

        with col1:
            # Always reload data from session state
            if not st.session_state.predictions_df.empty:
                # Add dynamic key to force chart updates
                chart_key = hash(tuple(st.session_state.predictions_df['timestamp']))
                
                latest_pred = st.session_state.predictions_df.iloc[-1]
                prediction = "Malignant" if latest_pred['predicted_label'] == 1 else "Benign"
                
                # Calculate confidence using the correct probability value
                confidence = latest_pred['malignant_probability'] if prediction == "Malignant" else latest_pred['benign_probability']
                confidence = confidence * 100  # Convert to percentage
                
                st.markdown(f"""
                    <div style="background: {'#e74c3c' if prediction == 'Malignant' else '#2ecc71'}; 
                        padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
                        <h3 style="color: white; margin: 0;">
                            Latest Prediction: {prediction}
                        </h3>
                        <p style="font-size: 1.2em; margin: 10px 0; color: white">
                            Confidence: {confidence:.1f}%
                        </p>
                        <p style="color: rgba(255,255,255,0.8); margin: 0;">
                            {latest_pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Visualization tabs with dynamic keys
                viz_tab1, viz_tab2, viz_tab3 = st.tabs([
                    "📊 Distribution & Evolution",
                    "📈 Timeline Analysis",
                    "🔍 Feature Analysis"
                ])
                
                with viz_tab1:
                    # Force cache invalidation by using the current timestamp
                    current_time = datetime.now().timestamp()
                    st.plotly_chart(
                        create_prediction_distribution_chart(st.session_state.predictions_df),
                        use_container_width=True,
                        key=f"dist_chart_{current_time}"  # Dynamic key based on time
                    )
                    st.plotly_chart(
                        create_prediction_evolution_chart(st.session_state.predictions_df),
                        use_container_width=True,
                        key=f"evo_chart_{current_time}"  # Dynamic key based on time
                    )
                
                with viz_tab2:
                    st.plotly_chart(
                        create_timeline_chart(st.session_state.predictions_df),
                        use_container_width=True,
                        key=f"timeline_{chart_key}"
                    )
                
                with viz_tab3:
                    # Group features by type
                    feature_groups = {
                        'Mean Values': [f for f in df.columns if '_mean' in f],
                        'Standard Error': [f for f in df.columns if '_se' in f],
                        'Worst Values': [f for f in df.columns if '_worst' in f]
                    }
                    
                    # Create feature selection
                    feature_type = st.selectbox(
                        "Select Feature Type",
                        options=list(feature_groups.keys())
                    )
                    
                    feature = st.selectbox(
                        "Select Feature for Analysis",
                        options=feature_groups[feature_type],
                        format_func=lambda x: x.replace('_', ' ').title()
                    )
                    
                    st.plotly_chart(
                        create_feature_histogram(st.session_state.predictions_df, feature),
                        use_container_width=True,
                        key=f"hist_{feature}_{chart_key}"
                    )
            
            else:
                st.info("👋 Waiting for predictions... Click 'Check New Predictions' to start monitoring.")

            # Get latest prediction
        #     if message:
        #         new_prediction['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #         if len(st.session_state.predictions) >= 100:  # Keep only last 100 predictions
        #             st.session_state.predictions.pop(0)
        #         st.session_state.predictions.append(new_prediction)
            
        #     # Display predictions
        #     for pred in reversed(st.session_state.predictions[-5:]):  # Show last 5 predictions
        #         prediction_class = "prediction-benign" if pred.get('predicted_label') == 0 else "prediction-malignant"
        #         label_text = "Benign" if pred.get('predicted_label') == 0 else "Malignant"
                
        #         st.markdown(f"""
        #             <div style="background: white; padding: 15px; border-radius: 10px; 
        #                         margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        #                 <div style="display: flex; justify-content: space-between; align-items: center;">
        #                     <span style="color: #666;">{pred.get('timestamp', 'N/A')}</span>
        #                     <span class="{prediction_class}">{label_text}</span>
        #                 </div>
        #                 <div style="margin-top: 10px;">
        #                     <div>Benign Probability: {pred.get('prob_benign', 0):.3f}</div>
        #                     <div style="background: #e9ecef; height: 8px; border-radius: 4px; margin: 5px 0;">
        #                         <div style="background: #00cc66; width: {pred.get('prob_benign', 0)*100}%; 
        #                                 height: 100%; border-radius: 4px;"></div>
        #                     </div>
        #                     <div>Malignant Probability: {pred.get('prob_malignant', 0):.3f}</div>
        #                     <div style="background: #e9ecef; height: 8px; border-radius: 4px; margin: 5px 0;">
        #                         <div style="background: #ff3366; width: {pred.get('prob_malignant', 0)*100}%; 
        #                                 height: 100%; border-radius: 4px;"></div>
        #                     </div>
        #                 </div>
        #             </div>
        #         """, unsafe_allow_html=True)
        
        # with col2:
        #     st.markdown("""
        #         <div style="background: linear-gradient(to right, #f8f9fa, #e9ecef);
        #                     padding: 20px; border-radius: 10px;">
        #             <h4>Statistics</h4>
        #         </div>
        #     """, unsafe_allow_html=True)
            
        #     total_predictions = len(st.session_state.predictions)
        #     benign_count = sum(1 for p in st.session_state.predictions if p.get('predicted_label') == 0)
        #     malignant_count = total_predictions - benign_count
            
        #     st.markdown(f"""
        #         <div style="background: white; padding: 20px; border-radius: 10px; 
        #                     box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;">
        #             <div style="margin-bottom: 20px;">
        #                 <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">
        #                     {total_predictions}
        #                 </div>
        #                 <div style="color: #666;">Total Predictions</div>
        #             </div>
        #             <div style="margin-bottom: 20px;">
        #                 <div style="font-size: 24px; font-weight: bold; color: #00cc66;">
        #                     {benign_count} ({(benign_count/total_predictions*100 if total_predictions > 0 else 0):.1f}%)
        #                 </div>
        #                 <div style="color: #666;">Benign Cases</div>
        #             </div>
        #             <div>
        #                 <div style="font-size: 24px; font-weight: bold; color: #ff3366;">
        #                     {malignant_count} ({(malignant_count/total_predictions*100 if total_predictions > 0 else 0):.1f}%)
        #                 </div>
        #                 <div style="color: #666;">Malignant Cases</div>
        #             </div>
        #         </div>
        #     """, unsafe_allow_html=True)
        
        # # Auto-refresh using empty placeholder
        # placeholder = st.empty()
        # with placeholder.container():
        #     st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # # Add manual refresh button
        # if st.button("Refresh"):
        #     st.rerun()

       
        with tab4:
                st.subheader("🔍 Advanced Data Insights")
                
                # Load the historical data
                df2 = pd.read_csv('/root/myproject/kafka_to_streamlit/predictions_history.csv')
                df2['timestamp'] = pd.to_datetime(df2['timestamp'])
                
                # Create tabs for different types of insights
                insight_tab1, insight_tab2, insight_tab3, insight_tab4= st.tabs([
                    "🎯 Prediction Accuracy Analysis",
                    "🔄 Pattern Recognition",
                    "📊 Feature Importance",
                    "📑 History File"
                ])
                
                with insight_tab1:
                    st.markdown("### Prediction Confidence Analysis")
                    fig_confidence = go.Figure()
                    
                    # scatter for benign
                    benign_sc = df2['predicted_label'] == 0
                    fig_confidence.add_trace(go.Scatter(
                        x = df2[benign_sc]['timestamp'],
                        y = df2[benign_sc]['benign_probability'],
                        mode='markers',
                        name='Benign Predictions',
                        marker = dict(color='#58CB16',size=8),
                        legendgroup='benign'
                    ))

                    # scatter for malignant
                    malignant_sc = df2['predicted_label']  == 1
                    fig_confidence.add_trace(go.Scatter(
                    x = df2[malignant_sc]['timestamp'],
                    y = df2[malignant_sc]['malignant_probability'],
                    mode='markers',
                    name='Malignant',
                    marker=dict(color='#EF2019',size=8),
                    legendgroup='Malignant'
                    ))
                    
                    fig_confidence.update_layout(
                        title='Prediction Confidence Over Time',
                        xaxis_title='Time',
                        yaxis_title = 'Probability',
                        height=400,
                        hovermode='x unified',
                        showlegend=True
                    )
                    st.plotly_chart(fig_confidence,use_container_width=True)
                    
                    # Add confidence distribution
                    col1, col2 = st.columns(2)
                    with col1:
                        benign_conf = df2[df2['predicted_label'] == 0]['benign_probability']
                        fig_benign = px.histogram(
                            benign_conf,
                            title='Benign Prediction Confidence Distribution',
                            color_discrete_sequence=['#2ecc71'],
                            labels={'value': 'Confidence', 'count': 'Frequency'}
                        )
                        st.plotly_chart(fig_benign, use_container_width=True)
                        
                    with col2:
                        malignant_conf = df2[df2['predicted_label'] == 1]['malignant_probability']
                        fig_malignant = px.histogram(
                            malignant_conf,
                            title='Malignant Prediction Confidence Distribution',
                            color_discrete_sequence=['#e74c3c'],
                            labels={'value': 'Confidence', 'count': 'Frequency'}
                        )
                        st.plotly_chart(fig_malignant, use_container_width=True)
                
                with insight_tab2:
                    st.markdown("### Temporal Pattern Analysis")
                    
                    # Time-based patterns
                    hourly_patterns = df2.groupby(df2['timestamp'].dt.hour)['predicted_label'].value_counts().unstack()
                    fig_hourly = px.bar(
                        hourly_patterns,
                        title='Prediction Distribution by Hour',
                        labels={'index': 'Hour of Day', 'value': 'Count'},
                        color_discrete_map={0: '#63CF1F', 1: '#E81414'},
                        barmode='group'
                    )
                    st.plotly_chart(fig_hourly, use_container_width=True)
                    
                    # Feature trends over time
                    selected_feature = st.selectbox(
                        "Select Feature to Analyze Trends",
                        options=[col for col in df2.columns if col.endswith(('_mean', '_se', '_worst'))]
                    )
                    
                    # fig_trend = px.scatter(
                    #     df2,
                    #     x='timestamp',
                    #     y=selected_feature,
                    #     color='predicted_label',
                    #     color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                    #     trendline="lowess",
                    #     title=f'Trend Analysis: {selected_feature}',
                    #     height=400
                    # )
                    
                    fig_trends = go.Figure()

                    fig_trends.add_trace(go.Scatter(
                        x = df2[benign_sc]['timestamp'],
                        y =   df2[benign_sc][selected_feature],
                        mode='markers',
                    
                        name='Benign',
                        marker=dict(color='#58CB16',size=8,line=dict(width=1,color='#45A110')),
                        showlegend=True
                    ))

                    fig_trends.add_trace(go.Scatter(
                        x = df2[malignant_sc]['timestamp'],
                        y = df2[malignant_sc][selected_feature],
                        mode='markers',
                        name='Malignant',
                        marker=dict(color='#EF2D19',size=8,line=dict(width=1,color='#CC1A08')),
                        showlegend=True
                    ))
                    fig_trends.update_layout(
                        title=f'Trend Analysis: {selected_feature}',
                        xaxis_title='Time',
                        yaxis_title=selected_feature.title(),
                        height=400,
                        hovermode='x unified',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ),
                        xaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.2)',
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='rgba(128, 128, 128, 0.2)',
                        )
                    )

                    st.plotly_chart(fig_trends, use_container_width=True)
                
                with insight_tab3:
                    st.markdown("### Feature Analysis Dashboard")
                    
                    # Feature correlation matrix
                    feature_cols = [col for col in df2.columns if col.endswith(('_mean', '_se', '_worst'))]
                    correlation_matrix = df2[feature_cols].corr()
                    
                    fig_corr = px.imshow(
                        correlation_matrix,
                        title="Feature Correlation Heatmap",
                        color_continuous_scale='RdBu_r',
                        aspect='auto'
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Improved Feature importance calculation
                    feature_correlations = []
                    for feature in feature_cols:
                        correlation = df2[feature].corr(df2['predicted_label'])
                        feature_correlations.append({
                            'feature': feature,
                            'correlation': abs(correlation),
                            'original_correlation': correlation
                        })
                    
                    feature_importance_df2 = pd.DataFrame(feature_correlations)
                    feature_importance_df2 = feature_importance_df2.sort_values('correlation', ascending=True)
                    
                    # Create improved feature importance visualization
                    fig_importance = go.Figure()
                    
                    # Add horizontal bar chart
                    fig_importance.add_trace(go.Bar(
                        y=feature_importance_df2['feature'],
                        x=feature_importance_df2['correlation'],
                        orientation='h',
                        marker=dict(
                            color=feature_importance_df2['original_correlation'],
                            colorscale='RdBu',
                            colorbar=dict(title="Correlation Direction"),
                            cmin=-1,
                            cmax=1
                        )
                    ))
                    
                    # Update layout for better readability
                    fig_importance.update_layout(
                        title="Feature Importance (Correlation with Prediction)",
                        xaxis_title="Absolute Correlation",
                        yaxis_title="Features",
                        height=800,  # Increase height for better readability
                        yaxis={'categoryorder': 'total ascending'},  # Sort bars by value
                        margin=dict(l=200),  # Increase left margin for feature names
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Add explanation of the visualization
                    st.markdown("""
                    #### Understanding Feature Importance:
                    - Bar length indicates the strength of correlation with the prediction
                    - Color indicates the direction of correlation:
                        - Red: Positive correlation (higher values associated with malignant)
                        - Blue: Negative correlation (higher values associated with benign)
                    - Longer bars indicate stronger relationships with the prediction outcome
                    """)
                    
                    # Interactive feature comparison (unchanged)
                    col1, col2 = st.columns(2)
                    with col1:
                        feature_x = st.selectbox("Select X-axis Feature", options=feature_cols, key='x_feature')
                    with col2:
                        feature_y = st.selectbox("Select Y-axis Feature", options=feature_cols, key='y_feature')
                    
                    fig_scatters = go.Figure()
                    fig_scatters.add_trace(go.Scatter(
                        x=df2[benign_sc][feature_x],
                        y=df2[benign_sc][feature_y],
                        mode='markers',
                        name='Benign',
                        marker=dict(color='#58CB16',size=8,line=dict(width=1,color='#45A110')))
                    )

                    fig_scatters.add_trace(go.Scatter(
                        x= df2[malignant_sc][feature_x],
                        y= df2[malignant_sc][feature_y],
                        mode='markers',
                        name='malignant',
                        marker=dict(color='#EF2D19',size=8,
                        line=dict(width=1,color='#CC1A08')))
                    )

                    fig_scatters.update_layout(
                        xaxis_title=feature_x,
                        yaxis_title=feature_y,
                        legend_title="Diagnosis"
                    )

                    st.plotly_chart(fig_scatters,use_container_width=True)
                with insight_tab4:
                    st.subheader("In this tab, You can see the predictions history")

                    st.dataframe(df2)

        with tab5:
            input_data = add_sidebar()
            st.markdown("<h6 id='tutorial'>Check This video To understand How</h6>",unsafe_allow_html=True)
            st.video("/root/myproject/kafka_to_streamlit/images/breast-cancer.mp4",loop=True,autoplay=True)
        


            with st.container():
                st.markdown("<h5>Try a case by selecting the value of each feature in the sidebar to see the prediction.</h5>",unsafe_allow_html=True)
                col1,col2 = st.columns([3,1])
                with col1:
                    radar = radar_chart(input_data)
                    st.plotly_chart(radar)
                with col2:
                
                    add_predictions(input_data,spark)
                    # st.write(prediction)
                    # st.success(probability)
if __name__ == "__main__":
    main()