import csv
import json
import os
import random
import time

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVICE_URL = os.getenv(
    "ML_SERVICE_URL",
    "http://ml_service:8000/api/prediction"
)

DATASET_PATH = os.getenv(
    "DATASET_PATH",
    "/app/train.csv"
)

RAM_SHIFT_PROB = float(os.getenv("RAM_SHIFT_PROB", "0"))

FEATURES = (
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
)

FLOAT_FEATURES = {"clock_speed", "m_dep"}


def load_dataset():
    with open(DATASET_PATH, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError("Dataset is empty")

    missing = set(FEATURES) - set(rows[0])

    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")

    return rows


def generate_payload(row):
    payload = {}

    for feature in FEATURES:
        value = row[feature]

        if feature in FLOAT_FEATURES:
            payload[feature] = float(value)
        else:
            payload[feature] = int(float(value))

    payload["pixel_area"] = (
        payload["px_height"] * payload["px_width"]
    )

    # Дополнительный режим для демонстрации data shift.
    if random.random() < RAM_SHIFT_PROB:
        payload["ram"] = random.randint(3500, 3998)

    return payload


def send_request(item_id, payload):
    data = json.dumps(payload).encode("utf-8")

    request = Request(
        url=f"{SERVICE_URL}?item_id={item_id}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=10) as response:
            result = json.loads(response.read())

            print(
                f"ID={item_id}, "
                f"RAM={payload['ram']}, "
                f"status={response.status}, "
                f"prediction={result}",
                flush=True,
            )

    except HTTPError as error:
        print(
            f"HTTP error: {error.code}, "
            f"body={error.read().decode(errors='replace')}",
            flush=True,
        )

    except URLError as error:
        print(f"Connection error: {error}", flush=True)


def main():
    rows = load_dataset()

    print(f"Loaded {len(rows)} samples", flush=True)
    print(f"Sending requests to {SERVICE_URL}", flush=True)

    item_id = 1

    while True:
        row = random.choice(rows)

        payload = generate_payload(row)

        send_request(item_id, payload)

        item_id += 1

        time.sleep(random.uniform(0, 5))


if __name__ == "__main__":
    main()
