# Proyecto de Detección de Características con OpenCV

Este proyecto implementa dos métodos de detección de características en imágenes utilizando OpenCV: **Local Binary Patterns (LBP)** y **Scale-Invariant Feature Transform (SIFT)**. Ambos archivos, `lbp.py` y `sift.py`, procesan imágenes y envían los resultados a través de WebSockets.

## Estructura del Proyecto

- `lbp.py`: Implementa la extracción de características usando LBP. Recibe una imagen, aplica el algoritmo de LBP y envía los resultados a través de WebSockets.
- `sift.py`: Implementa la detección de puntos clave con SIFT. Procesa una imagen y transmite los puntos clave detectados a través de WebSockets.

## Instalación

Asegúrate de tener Python instalado y ejecuta el siguiente comando para instalar las dependencias necesarias:

```sh
pip install opencv-python numpy websockets
