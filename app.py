# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO
import mysql.connector
from datetime import datetime
import json 
import time
import threading
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from scipy.fft import fft, fftfreq
from mysql.connector import Error
from datetime import datetime
from flask import Flask, request, send_file
import csv
import io


# Load the model without loading the optimizer configuration
my_model = load_model("modelResnet.h5", compile=False)

# Define the custom optimizer without weight decay
custom_optimizer = Adam()

# Compile the model with the custom optimizer
my_model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

data_vibration = []
app = Flask(__name__)
socketio = SocketIO(app, allow_unsafe_werkzeug=True)  # Add allow_unsafe_werkzeug=True


def insert_data(time_stamp, class_data):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='your_mysql_database',
            user='root',
            password='Praew_17'
        )

        if connection.is_connected():
            cursor = connection.cursor()
            # SQL query to insert data into the table
            query = "INSERT INTO predict_class (time_stamp, class) VALUES (%s, %s)"
            # Current timestamp
            current_time = datetime.now()
            # Tuple of values to insert
            data = (current_time, class_data)
            # Execute the query
            cursor.execute(query, data)
            # Commit changes
            connection.commit()
            print("Data inserted successfully.")
    except Error as e:
        print(f"Error inserting data: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")
            

def clear_screen():
    # Clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
def fft__(data):
    N = len(data)
    T = 1/10
    x1 = np.linspace(0.0, N*T, N, endpoint=False)
    y1 = np.array(data.astype(float))
    
    yf1 = fft(y1)
    xf1 = fftfreq(N, T)[:N//2]
    
    signals_FFT = [xf1, (2.0/N * np.abs(yf1[0:N//2]))]
    return signals_FFT

def pre_process(sent_data):
    # Convert sent_data to numpy array and perform FFT
    data_v1 = [row[0] for row in sent_data]
    data_v2 = [row[1] for row in sent_data]
    data_v3 = [row[2] for row in sent_data]
    # print(data_v2)
    clean_data_1 = np.array(data_v1, dtype=float)
    fft_result_1 = fft__(clean_data_1)
   
    clean_data_2 = np.array(data_v2, dtype=float)
    fft_result_2 = fft__(clean_data_2)
    
    clean_data_3 = np.array(data_v3, dtype=float)
    fft_result_3 = fft__(clean_data_3)

    # Extract frequency and amplitude pairs and create a list of dictionaries
    FFT_data_1 = [{'Hz': freq, 'value': amp} for freq, amp in zip(fft_result_1[0], fft_result_1[1])]
    FFT_data_2 = [{'Hz': freq, 'value': amp} for freq, amp in zip(fft_result_2[0], fft_result_2[1])]
    FFT_data_3 = [{'Hz': freq, 'value': amp} for freq, amp in zip(fft_result_3[0], fft_result_3[1])]
    socketio.emit('FFT_V1', FFT_data_1)
    socketio.emit('FFT_V2', FFT_data_2)
    socketio.emit('FFT_V3', FFT_data_3)
    
def pre(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Assuming ResNet50 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = my_model.predict(img_array)

    class_labels = ['Angular Misalignment', 'Combo', 'Normal', 'Parallel Misalignment', 'Unbalance and Angular Misalignment', 'Unbalance and Parallel Misalignment', 'Unbalance'] # Add your class labels here

    # Assuming predictions is a numpy array with shape (1, num_classes)
    # Where num_classes is the number of classes in your classification problem
    # Assuming my_model is your trained model

    # Decode the predictions
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    current_time = datetime.now()
    insert_data(current_time, predicted_class_label)
    # Print the predicted class label
    socketio.emit('Predicted_class', predicted_class_label)

# Connect to MySQL
def emit_data():
    while True:
        try:
            mydb = mysql.connector.connect(
                host="localhost",#"host.docker.internal",
                user="root",
                password="Praew_17",
                database="your_mysql_database"
            )
            
            mycursor = mydb.cursor()

            # Example: Select data from a table named 'mqtt_data'
            mycursor.execute("SELECT * FROM mqtt_data ORDER BY id DESC LIMIT 100")
            result = mycursor.fetchall()
            global data_vibration
            data = [{'id': row[0], 'timestamp': row[1].isoformat(), 'V1': row[2], 'V2': row[3], 'V3': row[4], 'SP': row[5], 'T': row[6]} for row in result]
            data_v= [row[2:5] for row in result]
            # Close cursor and MySQL connection
            mycursor.execute("SELECT * FROM predict_class ORDER BY time_stamp DESC LIMIT 100")
            class_ = mycursor.fetchall()
            
            mycursor.close()
            mydb.close()

            # Emit the data to the client
            socketio.emit('data', data)
            socketio.emit('pre_class', [{'timestamp': row[0].isoformat(),'class':row[1]} for row in class_])
            # Update the spectrogram
            signal_to_spectrogram(row[2] for row in result)
            pre_process(data_v)
            pre('static/spectrogram.jpg')
            clear_screen()
            
        except mysql.connector.Error as e:
            # Handle MySQL errors
            print("Error fetching data from MySQL:", e)
        time.sleep(0.01)

def signal_to_spectrogram(data):
    # Your code for generating spectrogram images
    # This is just a mock-up since I don't have your original code
    data_values_int = [int(value) for value in data]
   
    x = np.linspace(0, 10, 100)
    y = data_values_int

    
    # Create a new figure and axis
    fig, ax = plt.subplots()
    
    # Plot the spectrogram
    ax.specgram(y, Fs=10)
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
   
    
    # Add colorbar
    ax.specgram(y, Fs=10,scale='dB',cmap='jet',vmin=-5, vmax=30)

    # Save the plot to an image file
    image_path = 'static/spectrogram.jpg'
    plt.axis('off')
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig(image_path,bbox_inches='tight')

    # Emit the URL of the spectrogram image to the client
    image_url = f"/{image_path}"
    socketio.emit('spectrogram', image_url)
    plt.clf()
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/page3')
def contact():
    return render_template('page3.html')

@app.route('/download', methods=['GET', 'POST'])
def download():
    if request.method == 'POST':
        # Handle the POST request to download data
        selected_columns = request.form.getlist('columns')
        start_time = request.form.get('start')
        end_time = request.form.get('end')

        if not selected_columns:
            return "No columns selected", 400

        if not start_time or not end_time:
            return "Start and End times are required", 400

        columns_str = ', '.join(selected_columns)

        # Database connection
        connection = mysql.connector.connect(
            host='localhost',
            database='your_mysql_database',
            user='root',
            password='Praew_17'
        )
        cursor = connection.cursor()
        query = f"SELECT {columns_str} FROM mqtt_data WHERE created_at BETWEEN %s AND %s AND V1 IS NOT NULL AND V2 IS NOT NULL AND V3 IS NOT NULL"
        cursor.execute(query, (start_time, end_time))
        rows = cursor.fetchall()
        connection.close()

        # Create a BytesIO stream to hold the CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write the header
        writer.writerow(selected_columns)
        
        # Write the data rows
        for row in rows:
                writer.writerow(row)
            
            # Move to the beginning of the StringIO stream
        output.seek(0)

            # Convert StringIO to BytesIO for sending as a file
        buffer = io.BytesIO()
        buffer.write(output.getvalue().encode('utf-8'))
        buffer.seek(0)

        # Send the file as a download
        return send_file(buffer, as_attachment=True, download_name="data.csv", mimetype='text/csv')
    else:
        # Handle GET request to render the download form
        return render_template('download.html')


@socketio.on('connect')
def handle_connect():
    print('Client connected')

    # Start background threads to emit data and spectrogram images
    threading.Thread(target=emit_data, daemon=True).start()
    # threading.Thread(target=signal_to_spectrogram, args=(data,), daemon=True).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)
