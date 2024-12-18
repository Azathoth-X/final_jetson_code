import json
from fastapi import FastAPI,WebSocket,WebSocketDisconnect,Request,Response,status
import multiprocessing
# import logging
from fastapi.websockets import WebSocketState
from .readings import collect_data,save_collection_data
from contextlib import asynccontextmanager
from .schema import ResultInfoModel
import ipaddress
import os
import signal
import subprocess
from .inference import NP_ARRAYS_PATH, retrain_model

result_queue=multiprocessing.Queue()

@asynccontextmanager
async def lifespan(app:FastAPI):
    
    yield
    total_data_collected=len(os.listdir(NP_ARRAYS_PATH))

    if total_data_collected>= 100:
        retrain_model()
    
    subprocess.run(["shutdown", "-h"])
    return




app=FastAPI(debug=True,lifespan=lifespan)
connected_client = None
shutdownAvailable: bool=True


START_IP = ipaddress.ip_address("192.168.1.100")
END_IP = ipaddress.ip_address("192.168.1.200")


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global connected_client, shutdownAvailable
    reading_and_result = None  # Initialize process variable

    client_host=websocket.client.host

    try:
        host_ip=ipaddress.ip_address(client_host)
        if not START_IP<=host_ip<=END_IP:
            await websocket.close(code=1008,reason="IP not allowed")
            return
        if connected_client is not None:
            await websocket.close(code=1008, reason="Only one connection allowed.")
            return
                

    except ValueError:
        return
    

    connected_client = websocket
    await websocket.accept()
    shutdownAvailable=False

    try:
        
        while True:
           
            data = await websocket.receive_text()
            message = json.loads(data)

            if message is not None and message['type'] == 'START_ANALYSIS' and 'PREV_RESULT' not in message:

                patient_name:str = message['name']

                await websocket.send_text(json.dumps({'type': 'ANALYSIS_RESULT', 'result': 'starting'}))

                reading_and_result = multiprocessing.Process(target=collect_data,args=("test", patient_name,result_queue), daemon=False)

                reading_and_result.start()

                reading_and_result.join()

                reading_and_result.close()

                return_result:ResultInfoModel= result_queue.get()

                await websocket.send_text(return_result.model_dump_json())

                await websocket.close()   

                break

            if message is not None and 'PREV_RESULT' in message and message['type'] == 'START_ANALYSIS':

                patient_name:str=message['name']
                patient_results:bool=message['PREV_RESULT']

                await websocket.send_text(json.dumps({'type': 'ANALYSIS_RESULT', 'result': 'starting'}))

                reading_and_result = multiprocessing.Process(target=save_collection_data,args=("test", patient_name,result_queue,patient_results), daemon=False)

                reading_and_result.start()

                reading_and_result.join()

                reading_and_result.close()

                return_result:ResultInfoModel= result_queue.get()

                await websocket.send_text(return_result.model_dump_json())

                await websocket.close()

                break

    except WebSocketDisconnect:
        if reading_and_result and reading_and_result.is_alive():
            reading_and_result.kill()
            print("process killed")
            reading_and_result = None
        print("WebSocket connection closed")
        
    finally:
        reading_and_result = None
        shutdownAvailable=True
        connected_client = None
        if not websocket.application_state == WebSocketState.DISCONNECTED:
            await websocket.close(code=1000, reason="Connection closed in finally block.")


@app.get('/shutdown')
def shutdown_jetson(request:Request):
    client_ip = ipaddress.IPv4Address(str(request.client.host))
    if not START_IP<=client_ip<=END_IP:
        return Response(content=f'Host not allowed{client_ip}',status_code=status.HTTP_401_UNAUTHORIZED)
    if shutdownAvailable:
        os.kill(os.getpid(),signal.SIGINT)
        return Response("Shutting Down",status.HTTP_200_OK)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)