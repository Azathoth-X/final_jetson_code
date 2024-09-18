# #!/usr/bin/env python3

# import serial
# import time
# import pandas as pd
# import os
# import sys
# import threading
# from queue import Queue
# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaFileUpload
# from datetime import datetime

# # Re-execute the script with sudo if not running as root
# if os.geteuid() != 0:
#     print("This script must be run as root")
#     os.execvp("sudo", ["sudo"] + ["python3"] + sys.argv)

# # Define the serial ports for each Arduino
# arduino_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3']  # Adjust as needed
# baud_rate = 115200
# ser_list = []
# data_queue = Queue()
# stop_event = threading.Event()

# # Open serial connections
# for port in arduino_ports:
#     try:
#         ser = serial.Serial(port, baud_rate)
#         ser_list.append(ser)
#     except serial.SerialException as e:
#         print(f"Could not open port {port}: {e}")

# def read_from_port(ser, index):
#     while not stop_event.is_set():
#         try:
#             if ser.in_waiting > 0:
#                 line = ser.readline().decode('utf-8').strip()
#                 data_queue.put((index, float(line)))  # Add index to identify the port
#         except serial.SerialException as e:
#             print(f"Serial exception on port {index}: {e}")
#             break
#         except Exception as e:
#             data_queue.put((index, None))  # Append None if there's an error
#             print(f"Error reading from port {index}: {e}")
#         time.sleep(0.02)  # Adjust the sleep time as needed

# # Create and start a thread for each serial port
# threads = []
# for index, ser in enumerate(ser_list):
#     if ser.is_open:
#         thread = threading.Thread(target=read_from_port, args=(ser, index))
#         thread.daemon = True
#         thread.start()
#         threads.append(thread)

# # Collect data from the queue
# data = [[] for _ in range(len(arduino_ports))]
# limit = 400
# skip = 50

# try:
#     while all(len(d) < limit for d in data):  # Collect 3000 values for each port
#         while not data_queue.empty():
#             index, value = data_queue.get()
#             if len(data[index]) < limit:
#                 data[index].append(value)
#                 print(f'Port {index}: Value {value}')
#         time.sleep(0.01)  # Adjust the sleep time as needed

# except KeyboardInterrupt:
#     print("Program terminated.")
# finally:
#     stop_event.set()  # Signal threads to stop
#     for ser in ser_list:
#         if ser.is_open:
#             ser.close()

# # Wait for all threads to finish
# for thread in threads:
#     thread.join()

# # Ensure all 3000 values are collected
# for i, d in enumerate(data):
#     while len(d) < limit:
#         d.append(None)  # Fill with None if fewer than 3000 values collected

# # Transpose data to match the original format
# final_data = list(map(list, zip(*data)))

# # Create a DataFrame and save it to a CSV file
# df = pd.DataFrame(final_data, columns=['GO1', 'GO2', 'PANI_F1', 'Mg'])

# # Get inputs for folder and file name
# folder_name = input("Enter the folder name: ")
# file_name = input("Enter the file name: ")

# # Format the file name with current date and time
# current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# full_file_name = f"{file_name}_{current_time}.csv"

# # Save the DataFrame to a CSV file
# df.to_csv(full_file_name, index=False)

# print(f"Data collection completed and saved to {full_file_name}")

# # Extract the required range from the DataFrame
# df_extracted = df[skip:limit-skip]
# extracted_file_name = f"extracted_{full_file_name}"
# df_extracted.to_csv(extracted_file_name, index=False)

# # Path to your service account key file
# SERVICE_ACCOUNT_FILE = 'service_account.json'

# # Scopes for Google Drive API
# SCOPES = ['https://www.googleapis.com/auth/drive.file']

# # Authenticate using the service account
# credentials = service_account.Credentials.from_service_account_file(
#         SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# # Build the Drive API client
# drive_service = build('drive', 'v3', credentials=credentials)

# # Function to get or create a folder in Google Drive
# def get_or_create_folder(drive_service, folder_name, parent_id=None):
#     query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
#     if parent_id:
#         query += f" and '{parent_id}' in parents"

#     results = drive_service.files().list(q=query, fields="files(id, name)").execute()
#     items = results.get('files', [])

#     if not items:
#         file_metadata = {
#             'name': folder_name,
#             'mimeType': 'application/vnd.google-apps.folder'
#         }
#         if parent_id:
#             file_metadata['parents'] = [parent_id]

#         folder = drive_service.files().create(body=file_metadata, fields='id').execute()
#         return folder.get('id')
#     else:
#         return items[0]['id']

# # Use the function to get or create the folder
# folder_id = get_or_create_folder(drive_service, folder_name, '1OlfTRlnNXJx-KJw2AAlls9kaQ9-YzePn')

# # File metadata and media to be uploaded
# file_metadata = {
#     'name': extracted_file_name,
#     'parents': [folder_id]
# }
# media = MediaFileUpload(extracted_file_name, mimetype='text/csv')

# # Upload the file
# file = drive_service.files().create(body=file_metadata,
#                                     media_body=media,
#                                     fields='id').execute()

# print('File ID: %s' % file.get('id'))

# # Remove the local file after upload
# os.remove(extracted_file_name)
# os.remove(full_file_name)

# print("Local files removed")


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

# app = FastAPI()

# Define the serial ports for each Arduino
arduino_ports = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyACM2', '/dev/ttyACM3']  
baud_rate = 115200
ser_list = []
data_queue = Queue()
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

def collect_data(folder_name: str="test", file_name: str="test"):
    threads = []
    for index, ser in enumerate(ser_list):
        if ser.is_open:
            thread = threading.Thread(target=read_from_port, args=(ser, index))
            thread.daemon = True
            thread.start()
            threads.append(thread)

    # Collect data from the queue
    data = [[] for _ in range(len(arduino_ports))]
    limit = 1000
    skip = 50

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

    # Save the DataFrame to a CSV file
    df.to_csv(full_file_name, index=False)
    print(f"Data collection completed and saved to {full_file_name}")

    # Extract the required range from the DataFrame
    df_extracted = df[skip:limit - skip]
    extracted_file_name = f"extracted_{full_file_name}"
    df_extracted.to_csv(extracted_file_name, index=False)

    # Upload extracted file to Google Drive
    upload_to_drive(folder_name, extracted_file_name, full_file_name)
    return "complete"



# @app.post("/collect")
# async def start_data_collection(folder_name: str, file_name: str, background_tasks: BackgroundTasks):
#     """
#     Endpoint to trigger data collection.
#     - `folder_name`: The folder name in Google Drive
#     - `file_name`: The base name for the CSV file
#     """
#     # background_tasks.add_task(collect_data, folder_name, file_name)
#     return {"message": "Data collection started in the background"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# @app.add_websocket_route()