# Proyecto: Accomodation Predictor

## Requisitos

1. **Python 3.12**: Este proyecto requiere Python en su versión 3.12.
2. **Dependencias**: Las dependencias necesarias están especificadas en el archivo `requirements.txt`.

## Configuración

Sigue estos pasos para configurar y ejecutar el proyecto:

1. **Clonar el repositorio**:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```

2. **Crear un entorno virtual**:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar el proyecto**:
   Ejecuta el archivo `main.py` para iniciar el proyecto.
   ```bash
   python main.py
   ```

---

## Explicación del Código

### Descripción General

El código proporciona una clase llamada `AccomodationPredictor` que utiliza aprendizaje automático para predecir el tipo de alojamiento basado en datos de entrada. A continuación, se detalla cada parte del código:

### Componentes Principales

1. **Librerías Importadas**:
   - `pandas`: Para manipulación y análisis de datos.
   - `LabelEncoder` de `sklearn.preprocessing`: Para convertir datos categóricos en números.
   - `RandomForestClassifier` de `sklearn.ensemble`: Modelo de clasificación basado en árboles.

2. **Clase `AccomodationPredictor`**:
   - `__init__`: Inicializa el modelo, los codificadores de etiquetas y las columnas relevantes.
   - `load_training_data`: Lee y estructura los datos de entrenamiento desde un archivo de texto.
   - `preprocess_training_data`: Realiza preprocesamiento, como manejo de valores faltantes y codificación de datos categóricos.
   - `train_model`: Entrena el modelo usando los datos procesados.
   - `predict_from_csv`: Predice el tipo de alojamiento a partir de un archivo CSV con nuevos datos.
   - `save_predictions_to_csv`: Guarda las predicciones en un nuevo archivo CSV.

### Flujo de Ejecución

1. **Cargar Datos de Entrenamiento**:
   - Los datos se leen desde un archivo especificado.
   - Cada registro se separa y estructura para análisis.

2. **Preprocesar Datos**:
   - Se manejan valores faltantes.
   - Los datos categóricos se convierten a valores numéricos usando `LabelEncoder`.

3. **Entrenar el Modelo**:
   - El modelo `RandomForestClassifier` se ajusta a las características seleccionadas.

4. **Hacer Predicciones**:
   - Nuevos datos se procesan de manera similar.
   - El modelo predice el tipo de alojamiento.

5. **Guardar Resultados**:
   - Las predicciones se almacenan en un nuevo archivo CSV junto con los datos originales.

### Ejemplo de Uso

```python
from accomodation_predictor import AccomodationPredictor

# Inicializar predictor
predictor = AccomodationPredictor()

# Cargar y entrenar con datos
predictor.load_training_data("./data/train_data.txt")
predictor.preprocess_training_data()
predictor.train_model()

# Predecir con nuevos datos
data_path = "./data/new_data.csv"
predictions = predictor.predict_from_csv(data_path)

# Guardar predicciones
output_path = "./data/predictions_with_results.csv"
predictor.save_predictions_to_csv(data_path, predictions, output_path)
```

---

## Notas
- Asegúrate de que los archivos de datos estén en la ruta correcta antes de ejecutar el proyecto.
- Este proyecto usa un modelo de bosque aleatorio que puede ajustarse o reemplazarse según los requisitos específicos del usuario.

