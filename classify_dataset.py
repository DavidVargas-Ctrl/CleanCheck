# classify_dataset.py

import os
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Layer
from datetime import datetime

# Habilitar el crecimiento de memoria para la GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices):
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Crecimiento de memoria para GPU habilitado.")
    except Exception as e:
        print(f"Error al habilitar el crecimiento de memoria para GPU: {e}")

# Definir parámetros para el cargador de dataset
BATCH_SIZE = 32  # Puedes ajustar a 64 si tu GPU lo permite
IMG_SIZE = (224, 224)  # Tamaño esperado por MobileNetV2
N_CHANNELS = 3
IMG_SHAPE = IMG_SIZE + (N_CHANNELS,)

# Cambiar el número de clases a 2
N_CLASSES = 2

# Obtener variables de entorno que controlan la ejecución
model_name = os.getenv("HANDWASH_NN", "MobileNetV2")
num_trainable_layers = int(os.getenv("HANDWASH_NUM_LAYERS", 0))
num_epochs = int(os.getenv("HANDWASH_NUM_EPOCHS", 10))  # Reducir epochs para acelerar entrenamiento
num_frames = int(os.getenv("HANDWASH_NUM_FRAMES", 5))
suffix = os.getenv("HANDWASH_SUFFIX", "")
pretrained_model_path = os.getenv("HANDWASH_PRETRAINED_MODEL", "")
num_extra_layers = int(os.getenv("HANDWASH_EXTRA_LAYERS", 0))

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])


def freeze_model(model):
    if num_trainable_layers == 0:
        for layer in model.layers:
            layer.trainable = False
        return False
    elif num_trainable_layers > 0:
        for layer in model.layers[:-num_trainable_layers]:
            layer.trainable = False
        for layer in model.layers[-num_trainable_layers:]:
            layer.trainable = True
        return True
    else:
        # num_trainable_layers negativo, set all to trainable
        for layer in model.layers:
            layer.trainable = True
        return True


def get_preprocessing_function():
    if model_name == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    elif model_name == "InceptionV3":
        return tf.keras.applications.inception_v3.preprocess_input
    elif model_name == "Xception":
        return tf.keras.applications.xception.preprocess_input
    return None


def get_default_model():
    if model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
    elif model_name == "InceptionV3":
        base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')
    elif model_name == "Xception":
        base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')
    else:
        print("Unknown model name", model_name)
        exit(-1)

    training = freeze_model(base_model)

    # Construir el modelo
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = data_augmentation(inputs)
    x = get_preprocessing_function()(x)
    x = base_model(x, training=training)
    x = tf.keras.layers.GlobalAveragePooling2D()(x) if num_extra_layers else tf.keras.layers.Flatten()(x)
    for i in range(num_extra_layers):
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    print(model.summary())

    return model


# Definición de la clase MobileNetPreprocessingLayer
class MobileNetPreprocessingLayer(Layer):
    def __init__(self, **kwargs):
        super(MobileNetPreprocessingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MobileNetPreprocessingLayer, self).build(input_shape)

    def call(self, x):
        return (x / 127.5) - 1.0

    def compute_output_shape(self, input_shape):
        return input_shape


def fit_model(name, model, train_ds, val_ds, test_ds, weights_dict):
    # Callbacks para early stopping y guardar el modelo
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mc = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        filepath=f"{name}_{timestamp}.keras",  # Incluir timestamp en el nombre del archivo
        save_best_only=True,
        verbose=1
    )

    # Añadir otros callbacks si es necesario
    callbacks = [es, mc]

    print("Iniciando el entrenamiento del modelo...")
    history = model.fit(
        train_ds,
        epochs=num_epochs,
        validation_data=val_ds,
        class_weight=weights_dict,
        callbacks=callbacks
    )

    # Guardar el modelo final con timestamp
    model_filename = f"{name}_final-model_{timestamp}.keras"
    model.save(model_filename)
    print(f"Modelo guardado en {model_filename}")

    # Visualizar precisión con nombres de ejes
    plt.figure(figsize=(10, 6))
    plt.grid(True, axis="y")
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.legend(loc='lower right')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.ylim([0, 1])
    plt.title('Precisión de Entrenamiento y Validación')
    plt.savefig(f"accuracy-{name}.pdf", format="pdf")
    plt.close()
    print(f"Curva de precisión guardada en accuracy-{name}.pdf")

    measure_performance("validation", name, model, val_ds)
    del val_ds

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    result_str = f'Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy * 100:.2f}%\n'
    print(result_str)
    with open(f"results-{name}.txt", "a+") as f:
        f.write(result_str)

    measure_performance("test", name, model, test_ds)


def measure_performance(ds_name, name, model, ds, num_classes=N_CLASSES):
    matrix = [[0] * num_classes for _ in range(num_classes)]

    y_predicted = []
    y_true = []

    for images, labels in ds:
        predicted = model.predict(images, verbose=0)
        y_predicted.extend(np.argmax(predicted, axis=1))
        y_true.extend(labels.numpy())

    for y_p, y_t in zip(y_predicted, y_true):
        matrix[y_t][y_p] += 1

    print("Matriz de Confusión:")
    for row in matrix:
        print(row)

    f1_scores = []
    for i in range(num_classes):
        true_positives = matrix[i][i]
        predicted_positives = sum([matrix[j][i] for j in range(num_classes)])
        actual_positives = sum(matrix[i])
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Clase {i}: Precisión={precision * 100:.2f}% | Recall={recall * 100:.2f}% | F1 Score={f1:.2f}")
        f1_scores.append(f1)
    average_f1 = np.mean(f1_scores)
    s = f"F1 Score promedio en {ds_name}: {average_f1:.2f}\n"
    print(s)
    with open(f"results-{name}.txt", "a+") as f:
        f.write(s)


def evaluate(name, train_ds, val_ds, test_ds, weights_dict={}, model=None):
    name_with_suffix = name + suffix

    if pretrained_model_path:
        # Cargar y usar un modelo preentrenado
        custom_objects = {"MobileNetPreprocessingLayer": MobileNetPreprocessingLayer}
        try:
            base_model = tf.keras.models.load_model(pretrained_model_path, custom_objects=custom_objects)
            print("Modelo preentrenado cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo preentrenado: {e}")
            exit(-1)

        training = freeze_model(base_model)
        inputs = tf.keras.Input(shape=base_model.layers[0].output_shape[1:])
        # Ejecutar en modo inferencia
        outputs = base_model(inputs, training=training)
        model = tf.keras.Model(inputs, outputs)
        # Siempre entrenar la última capa
        model.layers[-1].trainable = True

        if "kaggle" in pretrained_model_path:
            name_with_suffix += "-pretrained-kaggle"
        elif "mitc" in pretrained_model_path:
            name_with_suffix += "-pretrained-mitc"
        elif "pskus" in pretrained_model_path:
            name_with_suffix += "-pretrained-pskus"
    else:
        # Crear un nuevo modelo
        if model is None:
            model = get_default_model()

    # Cambiar la función de pérdida a SparseCategoricalCrossentropy
    model.compile(
        optimizer='Adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    if num_extra_layers:
        name_with_suffix += f"-extralayers{num_extra_layers}"

    # Limpiar el archivo de resultados
    with open(f"results-{name_with_suffix}.txt", "w") as f:
        pass

    if pretrained_model_path:
        # Evaluar el modelo pre-entrenado antes del reentrenamiento
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
        result_str = f'Test loss antes del reentrenamiento: {test_loss:.4f} | Test accuracy: {test_accuracy * 100:.2f}%\n'
        print(result_str)
        with open(f"results-{name_with_suffix}.txt", "a+") as f:
            f.write(result_str)

        # Evaluar el rendimiento antes del reentrenamiento
        measure_performance("test-before-retraining", name_with_suffix, model, test_ds)

    # Llamar a la función fit_model sin el callback TerminateOnTime
    fit_model(name_with_suffix, model, train_ds, val_ds, test_ds, weights_dict)
