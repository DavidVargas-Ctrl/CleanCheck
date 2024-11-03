# real_time_analysis.py

import cv2
import tensorflow as tf
import numpy as np
from classify_dataset import get_preprocessing_function
from datetime import datetime

# Configuraciones
MODEL_PATH = 'kaggle-single-frame_final-model_20241102-214939.keras'  # Actualiza con el nombre correcto del modelo
IMG_SIZE = (224, 224)
N_CLASSES = 2
CONFIDENCE_THRESHOLD = 0.7  # Ajusta según sea necesario

# Cargar el modelo entrenado
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado exitosamente.")

# Obtener la función de preprocesamiento
preprocess_input = get_preprocessing_function()

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # Usa la cámara predeterminada. Cambia el índice si tienes múltiples cámaras.

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Ajustar la resolución de la cámara (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break

        # Preprocesar el frame
        input_image = cv2.resize(frame, IMG_SIZE)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = preprocess_input(input_image)
        input_image = np.expand_dims(input_image, axis=0)  # Añadir dimensión de batch

        # Realizar la predicción
        predictions = model.predict(input_image)
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)

        # Determinar la etiqueta y el color basado en la predicción
        if confidence > CONFIDENCE_THRESHOLD:
            if predicted_class == 0:
                label = "posición 1"
                color = (0, 255, 255)  # Amarillo en BGR
            elif predicted_class == 1:
                label = "posición 5"
                color = (0, 255, 0)  # Verde en BGR
            else:
                label = "Incierto"
                color = (255, 255, 0)  # Amarillo
        else:
            label = "no hay nada"
            color = (0, 0, 255)  # Rojo suave claro en BGR

        # Mostrar el label en el frame
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Si no se detecta nada, mostrar una pantalla roja suave
        if label == "no hay nada":
            overlay = frame.copy()
            overlay[:] = (0, 0, 255)  # Rojo
            alpha = 0.3  # Transparencia suave
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Mostrar el frame resultante
        cv2.imshow('Real-Time Movement Analysis', frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrumpido por el usuario.")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
