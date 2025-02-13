import cv2
import socket
import numpy as np


UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

original_image_path = "/home/anthony/Escritorio/Virtualenvs/env-SiftProject/src/data/cat.jpg"
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

resize_factor = 0.6
original_image_resized = cv2.resize(original_image, None, fx=resize_factor, fy=resize_factor)

sift = cv2.SIFT_create()

# Extraer keypoints y descriptores de la imagen original redimensionada
keypoints_original, descriptors_original = sift.detectAndCompute(original_image_resized, None)

# Redimensionar la imagen de superposición (ajusta el tamaño según sea necesario)
overlay_height, overlay_width = 1, 1
overlay_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
overlay_image = cv2.resize(overlay_image, (overlay_width, overlay_height))

# Extraer keypoints y descriptores de la imagen superpuesta
keypoints_overlay, descriptors_overlay = sift.detectAndCompute(overlay_image, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

def frame_generator():
    frame_buffer = bytearray()  # Buffer para almacenar los datos del frame
    buffer_size = 65536  # Tamaño del buffer para UDP

    while True:
        packet, _ = sock.recvfrom(buffer_size)  # Recibir paquete UDP
        frame_buffer.extend(packet)  # Agregar los datos al buffer

        # Intentar decodificar el frame cuando el buffer tiene suficientes datos
        try:
            frame = cv2.imdecode(np.frombuffer(frame_buffer, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame_buffer.clear()  # Limpiar el buffer después de procesar el frame

                # Convertir el frame a escala de grises para el procesamiento SIFT
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Obtener las dimensiones del frame
                frame_height, frame_width = frame_gray.shape

                # Calcular la posición para centrar la imagen superpuesta
                x_offset = (frame_width - overlay_width) // 2
                y_offset = (frame_height - overlay_height) // 2

                # Superponer la imagen en el frame
                frame_gray[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width] = overlay_image

                # Extraer keypoints y descriptores del frame actual (con la imagen superpuesta)
                keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

                # Emparejar las características entre la imagen original y el frame
                matches = bf.match(descriptors_original, descriptors_frame)

                # Ordenar los matches por distancia (mejores coincidencias primero)
                matches = sorted(matches, key=lambda x: x.distance)

                # Dibujar las coincidencias en el frame
                matched_frame = cv2.drawMatches(
                    original_image_resized, keypoints_original,  # Imagen original redimensionada y sus keypoints
                    frame_gray, keypoints_frame,                # Frame actual y sus keypoints
                    matches[:35], None,                         # Mostrar solo las 10 mejores coincidencias
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                cv2.imshow('Video con SIFT', matched_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Error al decodificar el frame: {e}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    frame_generator() 