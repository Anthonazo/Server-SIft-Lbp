from flask import Flask, Response
import cv2
import socket
import numpy as np
import json
import os

app = Flask(__name__)

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

def get_object_cascades(filename: str) -> dict:
    object_cascades = {}
    with open(filename, 'r') as fs:
        object_cascades = json.load(fs)
    if len(object_cascades.keys()) > 0:
        for object_cascade_name in object_cascades.keys():
            object_cascade_path = object_cascades[object_cascade_name]
            if os.path.exists(object_cascade_path):
                object_cascades[object_cascade_name] = cv2.CascadeClassifier(object_cascade_path)
                print(f"Cargado clasificador para: {object_cascade_name}")
            else:
                print(f"Error: No se encontró el archivo para {object_cascade_name} en {object_cascade_path}")
    else:
        raise ValueError('Load cascades into cascades.json.')
    return object_cascades

def frame_generator():
    frame_buffer = bytearray()
    buffer_size = 65536

    object_cascades = get_object_cascades('/home/anthony/Escritorio/Virtualenvs/env-SiftProject/src/cascades.json')
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        packet, _ = sock.recvfrom(buffer_size) 
        frame_buffer.extend(packet) 

        try:
            frame = cv2.imdecode(np.frombuffer(frame_buffer, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame_buffer.clear()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                for object_cascade_name in object_cascades.keys():
                    object_cascade = object_cascades[object_cascade_name]

                    # Parámetros específicos para cada clasificador
                    if object_cascade_name == "no_entry":
                        scaleFactor = 1.4  # Prueba con un valor más bajo
                        minNeighbors = 40   # Aumenta para reducir falsas detecciones
                    else:
                        scaleFactor = 1.4
                        minNeighbors = 30

                    objects = object_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(30, 30))

                    for (x, y, w, h) in objects:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        text = f"{object_cascade_name}"
                        cv2.putText(frame, text, (x + 5, y - 10), font, 0.9, (255, 255, 255), 2)


                cv2.imshow('Stream de Video', frame)

                # Esperar por la tecla 'q' para salir
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Error al decodificar el frame: {e}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    frame_generator()