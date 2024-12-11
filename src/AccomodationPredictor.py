import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class AccomodationPredictor:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.label_encoders = {}
        self.columns = ['durationOfStay', 'gender', 'Age', 'kids', 'destinationCode']

    def load_training_data(self, filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()

        records = []
        current_record = []

        for line in lines:
            cleaned_line = line.split("]")[1].strip().replace('"', '')
            if cleaned_line.startswith("Record"):
                if current_record:
                    records.append(current_record)
                current_record = []
            else:
                current_record.append(cleaned_line)

        if current_record:
            records.append(current_record)

        columns = ['id', 'durationOfStay', 'gender', 'Age', 'kids', 'destinationCode', 'AcomType']
        self.train_df = pd.DataFrame(records, columns=columns)

    def preprocess_training_data(self):
        for column in ['gender', 'destinationCode']:
            mode_value = self.train_df[column].mode()[0]
            self.train_df[column] = self.train_df[column].fillna(mode_value)

            le = LabelEncoder()
            le.fit(self.train_df[column])
            self.label_encoders[column] = le
            self.train_df[column] = le.transform(self.train_df[column])

        self.train_df['Age'] = pd.to_numeric(self.train_df['Age'], errors='coerce')
        self.train_df['durationOfStay'] = pd.to_numeric(self.train_df['durationOfStay'], errors='coerce')
        self.train_df['kids'] = pd.to_numeric(self.train_df['kids'], errors='coerce')

    def train_model(self):
        X_train = self.train_df[self.columns]
        y_train = self.train_df['AcomType']
        self.model.fit(X_train, y_train)

    def predict_from_csv(self, filepath):
        new_df = pd.read_csv(filepath)
        new_df = new_df.drop(columns='id')

        for column in ['gender', 'destinationCode']:
            mode_value = new_df[column].mode()[0]
            new_df[column] = new_df[column].fillna(mode_value)
            new_df[column] = self.label_encoders[column].transform(new_df[column])

        new_df['Age'] = pd.to_numeric(new_df['Age'], errors='coerce')
        new_df['durationOfStay'] = pd.to_numeric(new_df['durationOfStay'], errors='coerce')
        new_df['kids'] = pd.to_numeric(new_df['kids'], errors='coerce')

        new_df = new_df[self.columns]
        predictions = self.model.predict(new_df)
        return predictions
    
    def save_predictions_to_csv(self, input_path, predictions, output_path):
        """Guarda las predicciones en un nuevo archivo CSV"""
        # Leer el archivo CSV de entrada
        data = pd.read_csv(input_path)

        # Añadir las predicciones como una nueva columna
        data['Predictions'] = predictions

        # Guardar el DataFrame con las predicciones en un nuevo archivo CSV
        data.to_csv(output_path, index=False)
        print(f"Archivo con predicciones guardado en: {output_path}")

