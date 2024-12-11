from src.AccomodationPredictor import AccomodationPredictor

def main():
    try:
        predictor = AccomodationPredictor()

        # Cargar y preprocesar los datos de entrenamiento
        print("Cargando datos de entrenamiento...")
        predictor.load_training_data('data/train_data.txt')
        print("Preprocesando datos de entrenamiento...")
        predictor.preprocess_training_data()

        # Entrenar el modelo
        print("Entrenando el modelo...")
        predictor.train_model()

        # Hacer predicciones desde un archivo CSV
        print("Cargando archivo de predicciones...")
        predictions = predictor.predict_from_csv('data/TestDataAccomodation.csv')
        print("Predicciones realizadas:")
        for idx, prediction in enumerate(predictions, 1):
            print(f"Registro {idx}: {prediction}")
            
        # Guardar las predicciones en un nuevo archivo CSV
        predictor.save_predictions_to_csv('data\TestDataAccomodation.csv', predictions, './data/PredictionsWithResults.csv')

    except FileNotFoundError as e:
        print(f"Error: No se encontró un archivo necesario: {e}")
    except Exception as e:
        print(f"Se produjo un error inesperado: {e}")
        
if __name__ == "__main__":
    main()
