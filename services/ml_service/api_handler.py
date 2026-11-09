import pickle
 
from pathlib import Path
 
import pandas as pd
 
class FastAPIHandler:
 
    def __init__(self):
 
        # Путь внутри Docker-контейнера
        model_path = Path("/models/model.pkl")
 
        # Путь при локальном запуске
        if not model_path.exists():
            model_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "model.pkl"
            )
 
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)
 
    def predict(self, features: dict) -> int:
 
        # Преобразуем признаки в DataFrame
        data = pd.DataFrame([features])
 
        # Проверяем необходимые признаки
        if hasattr(self.model, "feature_names_in_"):
 
            expected = list(self.model.feature_names_in_)
 
            missing = [
                column
                for column in expected
                if column not in data.columns
            ]
 
            if missing:
                raise ValueError(
                    f"Не хватает признаков: {missing}"
                )
 
            # Сохраняем порядок признаков,
            # использовавшийся при обучении
            data = data[expected]
 
        # Получаем предсказание
        prediction = self.model.predict(data)
 
        return int(prediction[0])
