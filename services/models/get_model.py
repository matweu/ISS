from pathlib import Path
 
import mlflow
import mlflow.artifacts
 
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
 
# Run ID Production-модели из ЛР2
RUN_ID = "d804173f8a1949c99c5538419e7cccca"
 
def download_model():
 
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
 
    models_dir = Path(__file__).resolve().parent
 
    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=RUN_ID,
        artifact_path="model.pkl",
        dst_path=str(models_dir),
    )
 
    print("Модель успешно скачана!")
    print("Путь:", downloaded_path)
 
if __name__ == "__main__":
    download_model()
