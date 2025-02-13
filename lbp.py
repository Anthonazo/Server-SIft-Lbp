from flask import Flask, Response
import cv2
import socket
import numpy as np

app = Flask(__name__)

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

def frame_generator():
    frame_buffer = bytearray()
    buffer_size = 65536

    while True:
        packet, _ = sock.recvfrom(buffer_size)  # Recibir paquete UDP
        frame_buffer.extend(packet)  # Agregar los datos al buffer

        try:
            frame = cv2.imdecode(np.frombuffer(frame_buffer, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame_buffer.clear()  # Limpiar el buffer después de procesar el frame
                cv2.imshow('Stream de Video', frame)  # Mostrar el frame en una ventana

                # Esperar por la tecla 'q' para salir
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Error al decodificar el frame: {e}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    frame_generator()  