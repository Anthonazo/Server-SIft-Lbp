import cv2
import socket
import numpy as np

# Configuración de UDP
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
BUFFER_SIZE = 65536  

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

original_image_paths = {
    'cat': "/home/anthony/Escritorio/Virtualenvs/env-SiftProject/src/data/cat.jpg",
    'person': "/home/anthony/Escritorio/Virtualenvs/env-SiftProject/src/data/person.jpg"
}

category = 'person'  # Cambiar entre 'cat' y 'me'

original_image_path = original_image_paths[category]
original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

resize_factor = 0.6
original_image_resized = cv2.resize(original_image, None, fx=resize_factor, fy=resize_factor)

sift = cv2.SIFT_create()

keypoints_original, descriptors_original = sift.detectAndCompute(original_image_resized, None)

overlay_height, overlay_width = 1, 1
overlay_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
overlay_image = cv2.resize(overlay_image, (overlay_width, overlay_height))

keypoints_overlay, descriptors_overlay = sift.detectAndCompute(overlay_image, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

def frame_generator():
    frame_buffer = bytearray()

    while True:
        packet, _ = sock.recvfrom(BUFFER_SIZE)
        frame_buffer.extend(packet) 
        try:
            frame = cv2.imdecode(np.frombuffer(frame_buffer, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame_buffer.clear() 

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                frame_height, frame_width = frame_gray.shape

                x_offset = (frame_width - overlay_width) // 2
                y_offset = (frame_height - overlay_height) // 2

                frame_gray[y_offset:y_offset + overlay_height, x_offset:x_offset + overlay_width] = overlay_image

                keypoints_frame, descriptors_frame = sift.detectAndCompute(frame_gray, None)

                matches = bf.match(descriptors_original, descriptors_frame)

                matches = sorted(matches, key=lambda x: x.distance)

                matched_frame = cv2.drawMatches(
                    original_image_resized, keypoints_original,  
                    frame_gray, keypoints_frame,              
                    matches[:35], None,                   
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )

                category_matches = 0
                for m in matches:
                    if m.distance < 100:
                        category_matches += 1

                if category_matches > 10:
                    cv2.putText(matched_frame, f"Categoria detectada: {category} con {category_matches} coincidencias", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(matched_frame, "Categoria desconocida", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow('Video con SIFT', matched_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Error al decodificar el frame: {e}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    frame_generator()
