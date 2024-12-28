from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json
import uvicorn
from sklearn.preprocessing import MinMaxScaler
import logging
from fastapi.middleware.cors import CORSMiddleware

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    product_id: str
    days: int = 30

class PredictionResponse(BaseModel):
    product_id: str
    product_name: str
    predictions: List[Dict[str, float]]
    confidence_score: float

class EnhancedPharmacyDemandPredictor:
    def __init__(self, n_states=30, n_actions=30, learning_rate=0.1,
                 discount_factor=0.95, epsilon=0.1, window_size=30):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.window_size = window_size
        self.q_tables = {}  # Un Q-table por producto
        self.scalers = {}   # Un scaler por producto
        self.product_stats = {}  # Estadísticas por producto
        self.cached_predictions = {}  # Cache de predicciones
        self.cache_timeout = timedelta(hours=1)  # Tiempo de expiración del cache
        self.last_training_time = None
        self.data = None  # Cargar datos aquí
    
    def load_data(self, file_path: str):
        """
        Cargar el archivo CSV de ventas
        """
        try:
            self.data = pd.read_csv(file_path, parse_dates=['fecha_hora'])
            logger.info(f"Datos cargados desde {file_path}")
        except Exception as e:
            logger.error(f"Error al cargar los datos: {e}")
            raise
    
    def prepare_features(self, df, product_id):
        """
        Prepara características mejoradas para el modelo
        """
        try:
            # Filtrar por producto
            product_data = df[df['producto_id'] == product_id].copy()
            
            # Agregar ventas diarias
            daily_sales = product_data.groupby('fecha_hora').agg({
                'cantidad': 'sum',
                'total_venta': 'sum',
                'temperatura': 'mean',
                'humedad': 'first'
            }).reset_index()
            
            # Crear características estacionales
            daily_sales['dia_semana'] = daily_sales['fecha_hora'].dt.dayofweek
            daily_sales['mes'] = daily_sales['fecha_hora'].dt.month
            daily_sales['dia_mes'] = daily_sales['fecha_hora'].dt.day
            
            # Características cíclicas
            daily_sales['mes_sin'] = np.sin(2 * np.pi * daily_sales['mes'] / 12)
            daily_sales['mes_cos'] = np.cos(2 * np.pi * daily_sales['mes'] / 12)
            daily_sales['dia_semana_sin'] = np.sin(2 * np.pi * daily_sales['dia_semana'] / 7)
            daily_sales['dia_semana_cos'] = np.cos(2 * np.pi * daily_sales['dia_semana'] / 7)
            
            # Calcular características estadísticas
            features = pd.DataFrame({
                'fecha': daily_sales['fecha_hora'],
                'ventas_dia': daily_sales['cantidad'],
                'ventas_media_7d': daily_sales['cantidad'].rolling(7, min_periods=1).mean(),
                'ventas_media_30d': daily_sales['cantidad'].rolling(30, min_periods=1).mean(),
                'ventas_std_7d': daily_sales['cantidad'].rolling(7, min_periods=1).std().fillna(0),
                'variacion_diaria': daily_sales['cantidad'].pct_change().fillna(0),
                'temperatura': daily_sales['temperatura'],
                'es_humedo': (daily_sales['humedad'] == 'alta').astype(int),
                'dia_semana': daily_sales['dia_semana'],
                'mes': daily_sales['mes'],
                'mes_sin': daily_sales['mes_sin'],
                'mes_cos': daily_sales['mes_cos'],
                'dia_semana_sin': daily_sales['dia_semana_sin'],
                'dia_semana_cos': daily_sales['dia_semana_cos']
            })
            
            # Guardar estadísticas del producto
            self.product_stats[product_id] = {
                'max_ventas': daily_sales['cantidad'].max(),
                'media_ventas': daily_sales['cantidad'].mean(),
                'std_ventas': daily_sales['cantidad'].std(),
                'ultimo_valor': daily_sales['cantidad'].iloc[-1]
            }
            
            return features.fillna(0)
        except Exception as e:
            logger.error(f"Error al preparar las características: {e}")
            raise

    def train(self, features_df, product_id):
        """
        Entrena el modelo para un producto específico
        """
        try:
            if product_id not in self.q_tables:
                self.q_tables[product_id] = defaultdict(lambda: np.zeros(self.n_actions))
                
            total_reward = 0
            for i in range(len(features_df) - 1):
                current_state = self.get_state(features_df.iloc[i], product_id)
                next_state = self.get_state(features_df.iloc[i + 1], product_id)
                action = self.get_action(current_state, product_id)
                predicted_sales = (action / (self.n_actions - 1)) * self.product_stats[product_id]['max_ventas']
                actual_sales = features_df.iloc[i + 1]['ventas_dia']
                reward = self.get_reward(predicted_sales, actual_sales)
                total_reward += reward
                best_next_action = np.argmax(self.q_tables[product_id][next_state])
                current_q = self.q_tables[product_id][current_state][action]
                next_q = self.q_tables[product_id][next_state][best_next_action]
                new_q = current_q + self.learning_rate * (
                    reward + self.discount_factor * next_q - current_q
                )
                self.q_tables[product_id][current_state][action] = new_q
            return total_reward / len(features_df)
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {e}")
            raise

    def get_action(self, state, product_id):
        """
        Selecciona la acción basada en el estado actual usando la política epsilon-greedy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_tables[product_id][state])

    def get_state(self, features, product_id):
        """
        Calcula el estado basado en características normalizadas.
        """
        if product_id not in self.scalers:
            self.scalers[product_id] = MinMaxScaler()
        numeric_features = [
            'ventas_dia', 'ventas_media_7d', 'ventas_media_30d', 'ventas_std_7d',
            'variacion_diaria', 'temperatura', 'mes_sin', 'mes_cos',
            'dia_semana_sin', 'dia_semana_cos'
        ]
        features_norm = self.scalers[product_id].fit_transform(features[numeric_features].values.reshape(1, -1))
        state_value = np.mean(features_norm)
        return int(state_value * (self.n_states - 1))

    def save_model(self, filepath):
        """
        Guarda el modelo entrenado.
        """
        try:
            model_data = {
                'q_tables': {k: dict(v) for k, v in self.q_tables.items()},
                'product_stats': self.product_stats
            }
            with open(filepath, 'w') as f:
                json.dump(model_data, f)
            logger.info(f"Modelo guardado en {filepath}")
        except Exception as e:
            logger.error(f"Error al guardar el modelo: {e}")
            raise

    def load_model(self, filepath):
        """
        Carga el modelo desde un archivo.
        """
        try:
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            self.q_tables = {k: defaultdict(lambda: np.zeros(self.n_actions), v) for k, v in model_data['q_tables'].items()}
            self.product_stats = model_data['product_stats']
            logger.info(f"Modelo cargado desde {filepath}")
        except FileNotFoundError:
            logger.warning(f"El archivo {filepath} no existe.")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

predictor = EnhancedPharmacyDemandPredictor()

@app.on_event("startup")
def startup():
    try:
        predictor.load_data("ventas_farmacia_imperial.csv")
        predictor.load_model("modelo_entrenado.json")
    except Exception as e:
        logger.error(f"Error durante el inicio: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        features = predictor.prepare_features(predictor.data, request.product_id)
        predictions = predictor.predict_future(features, request.product_id, request.days)
        product_name = predictor.data[predictor.data['producto_id'] == request.product_id]['producto_nombre'].iloc[0]
        confidence_score = predictor.calculate_confidence_score(
            [p['demanda_predicha'] for p in predictions], request.product_id
        )
        return PredictionResponse(
            product_id=request.product_id,
            product_name=product_name,
            predictions=predictions,
            confidence_score=confidence_score
        )
    except Exception as e:
        logger.error(f"Error al realizar la predicción: {e}")
        raise HTTPException(status_code=500, detail="Error en la predicción")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
