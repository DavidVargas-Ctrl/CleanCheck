# kaggle-classify-frames.py

from classify_dataset import evaluate
from dataset_utilities import get_datasets

# Importar datetime para obtener el timestamp si es necesario
from datetime import datetime

# Asegúrate de proporcionar las rutas correctas a las carpetas en tu máquina
data_dir = r'C:\Users\USER\PycharmProjects\LavadoManos\kaggle-dataset-6classes-preprocessed\frames\trainval'
test_data_dir = r'C:\Users\USER\PycharmProjects\LavadoManos\kaggle-dataset-6classes-preprocessed\frames\test'

# Seleccionar solo las clases 'class_0' y 'class_5'
selected_classes = ['class_0', 'class_5']

# Cargar los datos y realizar la división
train_ds, val_ds, test_ds, weights_dict, class_names = get_datasets(data_dir, test_data_dir, selected_classes=selected_classes)

# Verificar que los conjuntos de datos no están vacíos
print(f"Entrenamiento: {len(train_ds)} batches")
print(f"Validación: {len(val_ds)} batches")
print(f"Prueba: {len(test_ds)} batches")
print(f"Clases seleccionadas: {class_names}")

# Ejecutar la evaluación del modelo
evaluate("kaggle-single-frame", train_ds, val_ds, test_ds, weights_dict)
