

#!/usr/bin/env python3

import serial
import time
import pandas as pd
# import os
# import sys
import threading
from queue import Queue
# from .files_handler import upload_to_drive
from datetime import datetime
# from fastapi import FastAPI, BackgroundTasks
# from typing import Optional
# from . import files_handler
from app.files_handler import upload_to_drive
# from app.app import prodModel


# Define the serial ports for each Arduino
arduino_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3']  
baud_rate = 115200
ser_list = []
data_queue = Queue()
result_queue = Queue()
stop_event = threading.Event()

# Open serial connections
for port in arduino_ports:
    try:
        ser = serial.Serial(port, baud_rate)
        ser_list.append(ser)
    except serial.SerialException as e:
        print(f"Could not open port {port}: {e}")

def read_from_port(ser, index):
    while not stop_event.is_set():
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                data_queue.put((index, float(line)))  # Add index to identify the port
        except serial.SerialException as e:
            print(f"Serial exception on port {index}: {e}")
            break
        except Exception as e:
            data_queue.put((index, None))  # Append None if there's an error
            print(f"Error reading from port {index}: {e}")
        time.sleep(0.02)  # Adjust the sleep time as needed

# def read_from_port(ser, index):
#     while not stop_event.is_set():
#         try:
#             if ser.is_open and ser.in_waiting > 0:  # Ensure serial port is open and has data
#                 line = ser.readline().decode('utf-8').strip()  # Read and decode data
#                 print(f"Port {index} received: {line}")
                
#                 if line:  # Ensure the line is not empty
#                     try:
#                         value = float(line)  # Try to convert to float
#                         data_queue.put((index, value))  # Add to queue with port index
#                     except ValueError:
#                         print(f"Non-numeric data received on port {index}: {line}")
#                 else:
#                     print(f"Empty line received from port {index}")
#             else:
#                 print(f"Port {index} not open or no data waiting")
#         except serial.SerialException as e:
#             print(f"Serial exception on port {index}: {e}")
#             break  # Exit loop on serial error
#         except Exception as e:
#             print(f"Error reading from port {index}: {e}")
#             break  # Exit loop on general error
#         time.sleep(0.02)  # Control the read frequency


def collect_data(folder_name: str, file_name:str):
    threads = []
    for index, ser in enumerate(ser_list):
        if ser.is_open:
            thread = threading.Thread(target=read_from_port, args=(ser, index))
            thread.daemon = True
            thread.start()
            threads.append(thread)

    # Collect data from the queue
    data = [[] for _ in range(len(arduino_ports))]
    limit = 100
    skip = 0

    try:
        while all(len(d) < limit for d in data):  # Collect 400 values for each port
            while not data_queue.empty():
                index, value = data_queue.get()
                if len(data[index]) < limit:
                    data[index].append(value)
                    print(f'Port {index}: Value {value}')
            time.sleep(0.01)  # Adjust the sleep time as needed

    except Exception as e:
        print(f"Error in data collection: {e}")
    finally:
        stop_event.set()  # Signal threads to stop
        for ser in ser_list:
            if ser.is_open:
                ser.close()

    for thread in threads:
        thread.join()

    # Ensure all 400 values are collected
    for i, d in enumerate(data):
        while len(d) < limit:
            d.append(None)  # Fill with None if fewer than 400 values collected

    # Transpose data to match the original format
    final_data = list(map(list, zip(*data)))

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(final_data, columns=['GO1', 'GO2', 'PANI_F1', 'Mg'])

    # Format the file name with current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_file_name = f"{file_name}_{current_time}.csv"

    # # Save the DataFrame to a CSV file
    df.to_csv(full_file_name, index=False)
    # print(f"Data collection completed and saved to {full_file_name}")

    # Extract the required range from the DataFrame
    # # df_extracted = df[skip:limit - skip]
    # extracted_file_name = f"extracted_{full_file_name}"
    # df.to_csv(extracted_file_name, index=False)

    # Upload extracted file to Google Drive
    upload_to_drive(folder_name, full_file_name)
    result_queue.put("done")



