# dataset_utilities.py

import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

def get_datasets(data_dir, test_data_dir, batch_size=32, selected_classes=['class_0', 'class_5']):
    # Asegúrate de que el batch_size sea 32 o menor
    if batch_size > 32:
        print(f"Reduciendo batch_size de {batch_size} a 32")
        batch_size = 32

    IMG_SIZE = (224, 224)  # Tamaño esperado por MobileNetV2
    BATCH_SIZE = batch_size

    # Función para recopilar rutas de imágenes y etiquetas
    def get_image_paths_and_labels(directory, selected_classes):
        image_paths = []
        labels = []
        class_names = sorted([d.name for d in Path(directory).iterdir() if d.is_dir() and d.name in selected_classes])
        class_indices = {name: index for index, name in enumerate(class_names)}
        for class_name in class_names:
            class_dir = Path(directory) / class_name
            # Recopilar imágenes para la clase actual
            class_image_paths = []
            for ext in ['.jpg', '.jpeg', '.png']:
                class_image_paths.extend([str(p) for p in class_dir.rglob(f'*{ext}')])
                class_image_paths.extend([str(p) for p in class_dir.rglob(f'*{ext.upper()}')])  # Soporte para mayúsculas
            # Extender la lista principal
            image_paths.extend(class_image_paths)
            # Extender las etiquetas únicamente con el número de imágenes de la clase actual
            labels.extend([class_indices[class_name]] * len(class_image_paths))
        return image_paths, labels, class_names

    # Cargar conjunto de prueba
    test_image_paths, test_labels, test_class_names = get_image_paths_and_labels(test_data_dir, selected_classes)

    # Cargar y dividir trainval
    trainval_image_paths, trainval_labels, trainval_class_names = get_image_paths_and_labels(data_dir, selected_classes)

    # Asegurarse de que las clases coincidan
    if trainval_class_names != test_class_names:
        raise ValueError("Las clases en trainval y test no coinciden.")

    # Dividir trainval en train y val
    train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
        trainval_image_paths, trainval_labels, test_size=0.2, random_state=42, stratify=trainval_labels)

    # Crear datasets TensorFlow
    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        # Utiliza la función de preprocesamiento adecuada según el modelo
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    # Función para crear un tf.data.Dataset
    def create_dataset(image_paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        ds = ds.cache()  # Cachear los datos en memoria
        ds = ds.shuffle(buffer_size=1000)  # Mezclar los datos para mejorar el entrenamiento
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(buffer_size=AUTOTUNE)  # Prefetch para optimizar la ejecución
        return ds

    train_ds = create_dataset(train_image_paths, train_labels)
    val_ds = create_dataset(val_image_paths, val_labels)
    test_ds = create_dataset(test_image_paths, test_labels)

    # Verificaciones de consistencia
    assert len(train_image_paths) == len(train_labels), "Mismatch between train_image_paths and train_labels"
    assert len(val_image_paths) == len(val_labels), "Mismatch between val_image_paths and val_labels"
    assert len(test_image_paths) == len(test_labels), "Mismatch between test_image_paths and test_labels"

    # Calcular pesos de clases para manejar desbalanceo
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    weights = {cls: total / count for cls, count in class_counts.items()}

    # Mapear etiquetas a nombres de clases
    weights_dict = {cls: weights[cls] for cls in weights}

    print("weights_dict=", weights_dict)
    print(f"Total imágenes en trainval: {len(trainval_image_paths)}")
    print(f"Total imágenes en test: {len(test_image_paths)}")
    print(f"Entrenamiento: {len(train_ds)} batches")
    print(f"Validación: {len(val_ds)} batches")
    print(f"Prueba: {len(test_ds)} batches")

    return train_ds, val_ds, test_ds, weights_dict, test_class_names
