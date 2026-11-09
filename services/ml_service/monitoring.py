import json
import time

from prometheus_client import Counter, Histogram, make_asgi_app
from main import app as ml_app


HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Number of HTTP requests",
    ["method", "path", "status"],
)

HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP response duration in seconds",
    ["path"],
    buckets=(
        0.005, 0.01, 0.025, 0.05, 0.1,
        0.25, 0.5, 1, 2.5, 5,
    ),
)

MODEL_PREDICTION = Histogram(
    "model_prediction",
    "Distribution of predicted mobile price categories",
    buckets=(0, 1, 2, 3),
)

MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Predictions by price range",
    ["price_range"],
)

MODEL_INPUT_RAM = Histogram(
    "model_input_ram",
    "Distribution of incoming RAM values in MB",
    buckets=(512, 1024, 1536, 2048, 3072, 4096),
)


for cls in range(4):
    MODEL_PREDICTIONS.labels(price_range=str(cls))


metrics_app = make_asgi_app()


async def app(scope, receive, send):
    if scope["type"] != "http":
        await ml_app(scope, receive, send)
        return

    path = scope.get("path", "")

    # Endpoint для Prometheus.
    if path == "/metrics":
        await metrics_app(scope, receive, send)
        return

    label_path = (
        path if path in ("/", "/api/prediction") else "other"
    )

    status = 500
    started = time.perf_counter()

    body_in = bytearray()
    body_out = bytearray()

    async def receive_recorded():
        message = await receive()

        if (
            path == "/api/prediction"
            and message["type"] == "http.request"
        ):
            chunk = message.get("body", b"")

            if len(body_in) < 65536:
                body_in.extend(
                    chunk[:65536 - len(body_in)]
                )

        return message

    async def send_recorded(message):
        nonlocal status

        if message["type"] == "http.response.start":
            status = message["status"]

        elif (
            message["type"] == "http.response.body"
            and path == "/api/prediction"
            and 200 <= status < 300
        ):
            body_out.extend(message.get("body", b""))

        await send(message)

    try:
        await ml_app(
            scope,
            receive_recorded,
            send_recorded,
        )

    finally:
        HTTP_REQUESTS.labels(
            scope["method"],
            label_path,
            str(status),
        ).inc()

        HTTP_DURATION.labels(label_path).observe(
            time.perf_counter() - started
        )

        if path == "/api/prediction" and 200 <= status < 300:
            try:
                response_data = json.loads(body_out)
                request_data = json.loads(body_in)

                prediction = int(response_data["predict"])
                ram = float(request_data["ram"])

            except (
                ValueError,
                KeyError,
                TypeError,
                UnicodeDecodeError,
            ):
                pass

            else:
                if 0 <= prediction <= 3:
                    MODEL_PREDICTION.observe(prediction)

                    MODEL_PREDICTIONS.labels(
                        price_range=str(prediction)
                    ).inc()

                MODEL_INPUT_RAM.observe(ram)
