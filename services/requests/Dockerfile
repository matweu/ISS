FROM python:3.11-slim

WORKDIR /app

COPY requests.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "requests.py"]

# Сборка образа:
# docker build -t mobile-price-requests:1 .

# Запуск образа:
# docker run --rm --network host \
#   -e ML_SERVICE_URL=http://127.0.0.1:8000/api/prediction \
#   -v "$(pwd)/../../data/train.csv:/app/train.csv:ro" \
#   mobile-price-requests:1
