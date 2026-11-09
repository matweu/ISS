FROM python:3.11-slim

COPY . /ml_service

VOLUME ["/models"]

WORKDIR /ml_service

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# Build:
# docker build -t mobile-price-service:1 .

# Run:
# docker run -d --name mobile-price-service -p 8000:8000 -v "$(pwd)/../models:/models:ro" mobile-price-service:1
